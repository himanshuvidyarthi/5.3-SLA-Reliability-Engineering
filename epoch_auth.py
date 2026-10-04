"""
epoch_auth.py — EPOCH Action Isolation Layer
Session 15.2. Step 2 of the three-step capstone.

Builds on epoch_ingest.py (15.1):
    15.1 → data isolation (per-hospital FAISS, composite checkpointer, RLS)
    15.2 → action isolation (JWT, ROLE_TOOLS, BoundGeminiModel, audit)

Components:
    - mint_jwt / verify_jwt: self-contained HMAC-SHA256 JWT (no external JWT
      dependency needed for the teaching demo). Claims carry hospital_id,
      sub (user id), and role — the three facts every downstream isolation
      decision is made from.
    - ROLE_TOOLS: the RBAC matrix. A role that isn't listed gets zero tools.
      Deny-by-default, same principle as 15.1's RLS and 15.3's timeout->504.
    - TOOL_REGISTRY: the tools an agent may be bound to, each simulated.
    - BoundGeminiModel: wraps a Gemini model object and restricts the tool
      list actually bound to it to ROLE_TOOLS[role] — a role can't even see
      a tool schema it isn't entitled to, let alone call it.
    - audit_tool_call / AUDIT_LOG: every tool invocation is appended to an
      in-memory audit trail with actor, hospital, tool, and timestamp.
    - trace_tags: a lightweight context manager standing in for OTel span
      attributes, so call sites read the way the production tracing code
      will read once real OTel is wired in.
    - build_agent_for_role / epoch_invoke: the 15.2 gateway — JWT verify,
      then build a role-bound agent, then run it. epoch_sla.py (15.3) wraps
      this same shape with a hard deadline and a circuit breaker.

Design mandate:
    A ROLE CANNOT USE A TOOL IT WAS NEVER BOUND. Restricting at the prompt
    level ("please don't call this tool") is not isolation. Isolation means
    the tool schema is never in the bound tool list in the first place.
"""

import base64
import contextlib
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from epoch_ingest import router as faiss_router, seed_corpus

# Loads .env if present; a no-op if it isn't. Everything below has a working
# default, so the module runs identically with or without a .env file.
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: JWT constants + mint/verify (self-contained HMAC-SHA256)
# ─────────────────────────────────────────────────────────────────────────────

JWT_SECRET   = os.environ.get('JWT_SECRET') or 'epoch-teaching-secret-do-not-use-in-prod'
JWT_ALG      = 'HS256'
JWT_ISSUER   = 'epoch-gateway'
JWT_AUDIENCE = 'epoch-clients'
JWT_TTL_SECONDS = 900  # 15 minutes


class AuthError(Exception):
    """Raised for any JWT or authorization failure. Carries a machine-
    readable `code` and a human `detail` so callers can branch on `code`
    without string-matching `detail`."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f'{code}: {detail}')


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def _b64url_decode(data: str) -> bytes:
    padding = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(header_b64: str, payload_b64: str) -> str:
    msg = f'{header_b64}.{payload_b64}'.encode('ascii')
    sig = hmac.new(JWT_SECRET.encode('ascii'), msg, hashlib.sha256).digest()
    return _b64url_encode(sig)


def mint_jwt(hospital_id: str, user_id: str, role: str, ttl_seconds: int = JWT_TTL_SECONDS) -> str:
    """
    Mint a signed JWT carrying hospital_id, sub, and role.
    Introduced: Session 15.2. Permanent.
    """
    now = int(time.time())
    header = {'alg': JWT_ALG, 'typ': 'JWT'}
    payload = {
        'iss': JWT_ISSUER,
        'aud': JWT_AUDIENCE,
        'sub': user_id,
        'hospital_id': hospital_id,
        'role': role,
        'iat': now,
        'exp': now + ttl_seconds,
        'jti': uuid.uuid4().hex,
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(',', ':')).encode('utf-8'))
    sig_b64 = _sign(header_b64, payload_b64)
    return f'{header_b64}.{payload_b64}.{sig_b64}'


def verify_jwt(token: str) -> Dict[str, Any]:
    """
    Verify signature, issuer, audience, and expiry. Returns claims on success.
    Raises AuthError on any failure — deny by default.
    Introduced: Session 15.2. Permanent.
    """
    parts = token.split('.')
    if len(parts) != 3:
        raise AuthError('malformed_token', 'Token does not have three segments.')
    header_b64, payload_b64, sig_b64 = parts

    expected_sig = _sign(header_b64, payload_b64)
    if not hmac.compare_digest(expected_sig, sig_b64):
        raise AuthError('bad_signature', 'JWT signature verification failed.')

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as e:
        raise AuthError('malformed_payload', f'Could not decode JWT payload: {e}')

    if payload.get('iss') != JWT_ISSUER:
        raise AuthError('bad_issuer', f"Expected issuer {JWT_ISSUER!r}, got {payload.get('iss')!r}.")
    if payload.get('aud') != JWT_AUDIENCE:
        raise AuthError('bad_audience', f"Expected audience {JWT_AUDIENCE!r}, got {payload.get('aud')!r}.")
    if payload.get('exp', 0) < time.time():
        raise AuthError('token_expired', 'Token expiry has passed.')
    if not payload.get('hospital_id') or not payload.get('role') or not payload.get('sub'):
        raise AuthError('missing_claims', 'Token is missing hospital_id, role, or sub.')

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ROLE_TOOLS RBAC matrix + TOOL_REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def _tool_lookup_patient_history(patient_ref: str, **_) -> dict:
    return {'tool': 'lookup_patient_history', 'patient_ref': patient_ref,
            'records': 3, 'summary': 'Non-smoker, no prior malignancy.'}


def _tool_fetch_prior_imaging(patient_ref: str, **_) -> dict:
    return {'tool': 'fetch_prior_imaging', 'patient_ref': patient_ref,
            'prior_ct_count': 1, 'most_recent': '2024-03-12'}


def _tool_search_expert_corpus(hospital_id: str, expert_id: str, query: str, **_) -> dict:
    hits = faiss_router.search(hospital_id, expert_id, query)
    return {'tool': 'search_expert_corpus', 'hits': hits}


def _tool_prescribe_followup(patient_ref: str, plan: str, **_) -> dict:
    return {'tool': 'prescribe_followup', 'patient_ref': patient_ref, 'plan': plan, 'status': 'ordered'}


TOOL_REGISTRY: Dict[str, Callable[..., dict]] = {
    'lookup_patient_history': _tool_lookup_patient_history,
    'fetch_prior_imaging':    _tool_fetch_prior_imaging,
    'search_expert_corpus':   _tool_search_expert_corpus,
    'prescribe_followup':     _tool_prescribe_followup,
}

# Deny-by-default: a role not present here gets zero tools.
ROLE_TOOLS: Dict[str, List[str]] = {
    'viewer':  ['search_expert_corpus'],
    'analyst': ['search_expert_corpus', 'lookup_patient_history', 'fetch_prior_imaging'],
    'admin':   ['search_expert_corpus', 'lookup_patient_history', 'fetch_prior_imaging', 'prescribe_followup'],
}

MODEL_UTILITY  = 'gemini-2.5-flash'
MODEL_FRONTIER = 'gemini-2.5-pro'


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: BoundGeminiModel — tool binding scoped to ROLE_TOOLS
# ─────────────────────────────────────────────────────────────────────────────

class BoundGeminiModel:
    """
    Wraps a model identifier and restricts its bound tool list to exactly
    ROLE_TOOLS[role]. A role can't call a tool it isn't bound — the tool
    schema for anything outside its allowlist is never attached.

    Production: genai.GenerativeModel(model_name, tools=[...allowed specs]).
    Teaching: stores the allowed tool names and model name; `.invoke_tool`
    enforces the allowlist before dispatching into TOOL_REGISTRY.

    Introduced: Session 15.2. Permanent.
    """

    def __init__(self, model_name: str, role: str, hospital_id: str, user_id: str):
        self.model_name  = model_name
        self.role        = role
        self.hospital_id = hospital_id
        self.user_id     = user_id
        self.allowed_tools: List[str] = ROLE_TOOLS.get(role, [])

    def invoke_tool(self, tool_name: str, **kwargs) -> dict:
        if tool_name not in self.allowed_tools:
            raise AuthError(
                'tool_not_bound',
                f"Role {self.role!r} is not bound to tool {tool_name!r}. "
                f"Allowed: {self.allowed_tools}",
            )
        fn = TOOL_REGISTRY[tool_name]
        result = fn(hospital_id=self.hospital_id, **kwargs)
        audit_tool_call(self.user_id, self.hospital_id, tool_name, result)
        return result


def build_agent_for_role(role: str, hospital_id: str, user_id: str,
                          model_name: str = MODEL_UTILITY) -> BoundGeminiModel:
    """Constructs a BoundGeminiModel scoped to the caller's role and tenant.
    Introduced: Session 15.2. Permanent."""
    return BoundGeminiModel(model_name=model_name, role=role, hospital_id=hospital_id, user_id=user_id)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Audit trail
# ─────────────────────────────────────────────────────────────────────────────

AUDIT_LOG: List[Dict[str, Any]] = []


def audit_tool_call(user_id: str, hospital_id: str, tool_name: str, result: Any) -> None:
    """Append one immutable audit entry. Introduced: Session 15.2. Permanent."""
    AUDIT_LOG.append({
        'ts': time.time(),
        'user_id': user_id,
        'hospital_id': hospital_id,
        'tool_name': tool_name,
        'result_summary': str(result)[:200],
    })


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: trace_tags — OTel-style span attribute context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def trace_tags(**tags):
    """
    Stand-in for `with tracer.start_as_current_span(...) as span:
    span.set_attributes(tags)`. Teaching version just records tags onto a
    module-level list so tests can assert on them; production swaps this
    for a real OTel span context manager with the same call shape.
    Introduced: Session 15.2. Permanent.
    """
    entry = {'tags': tags, 'ts': time.time()}
    _TRACE_SPANS.append(entry)
    try:
        yield entry
    finally:
        entry['duration_s'] = time.time() - entry['ts']


_TRACE_SPANS: List[Dict[str, Any]] = []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: epoch_invoke — the 15.2 auth-gated gateway
# ─────────────────────────────────────────────────────────────────────────────

def epoch_invoke(authorization_header: str, body: dict) -> dict:
    """
    JWT-verified, role-bound gateway. Verifies the bearer token, builds a
    BoundGeminiModel for the caller's role, and (for this teaching demo)
    performs one illustrative tool call to prove role-scoping end to end.

    epoch_sla.epoch_invoke_sla (15.3) wraps this same shape with a hard
    deadline and a circuit breaker.

    Introduced: Session 15.2. Permanent.
    """
    if not authorization_header.startswith('Bearer '):
        return {'status': 401, 'error': 'missing_bearer'}
    token = authorization_header[7:]
    try:
        claims = verify_jwt(token)
    except AuthError as e:
        return {'status': 401, 'error': e.code, 'detail': e.detail}

    hospital_id = claims['hospital_id']
    role        = claims['role']
    user_id     = claims['sub']
    tool_name   = body.get('tool_name', 'search_expert_corpus')
    tool_args   = {k: v for k, v in body.items() if k != 'tool_name'}
    tool_args.setdefault('expert_id', 'chen')
    tool_args.setdefault('query', body.get('question', ''))
    tool_args.setdefault('patient_ref', 'patient-demo')

    with trace_tags(hospital_id=hospital_id, role=role, tool_name=tool_name):
        agent = build_agent_for_role(role, hospital_id, user_id)
        try:
            result = agent.invoke_tool(tool_name, **tool_args)
        except AuthError as e:
            return {'status': 403, 'error': e.code, 'detail': e.detail}

    return {'status': 200, 'hospital_id': hospital_id, 'role': role, 'result': result}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Verification
# ─────────────────────────────────────────────────────────────────────────────

def run_session_verification() -> dict:
    checks = []
    seed_corpus()

    # CHECK 1: mint/verify roundtrip
    tok = mint_jwt('hospital_a', 'dr-jones', 'analyst')
    claims = verify_jwt(tok)
    checks.append({
        'label': 'mint_jwt / verify_jwt roundtrip preserves claims',
        'passed': claims['hospital_id'] == 'hospital_a' and claims['role'] == 'analyst' and claims['sub'] == 'dr-jones',
        'note': f'claims={ {k: claims[k] for k in ("hospital_id","role","sub")} }',
    })

    # CHECK 2: expired token rejected
    expired_tok = mint_jwt('hospital_a', 'dr-jones', 'analyst', ttl_seconds=-10)
    raised = False
    try:
        verify_jwt(expired_tok)
    except AuthError as e:
        raised = e.code == 'token_expired'
    checks.append({
        'label': 'verify_jwt rejects expired tokens',
        'passed': raised,
        'note': f'rejected={raised}',
    })

    # CHECK 3: tampered signature rejected
    tampered = tok[:-4] + ('A' if tok[-4] != 'A' else 'B') + tok[-3:]
    raised2 = False
    try:
        verify_jwt(tampered)
    except AuthError as e:
        raised2 = e.code == 'bad_signature'
    checks.append({
        'label': 'verify_jwt rejects a tampered signature',
        'passed': raised2,
        'note': f'rejected={raised2}',
    })

    # CHECK 4: ROLE_TOOLS deny-by-default — viewer cannot call prescribe_followup
    viewer_agent = build_agent_for_role('viewer', 'hospital_a', 'nurse-lee')
    denied = False
    try:
        viewer_agent.invoke_tool('prescribe_followup', patient_ref='p1', plan='x')
    except AuthError as e:
        denied = e.code == 'tool_not_bound'
    checks.append({
        'label': 'ROLE_TOOLS denies a role a tool outside its allowlist',
        'passed': denied,
        'note': f'viewer denied prescribe_followup={denied}',
    })

    # CHECK 5: admin can call every registered tool
    admin_agent = build_agent_for_role('admin', 'hospital_a', 'dr-admin')
    admin_ok = set(admin_agent.allowed_tools) == set(TOOL_REGISTRY.keys())
    checks.append({
        'label': 'admin role is bound to every tool in TOOL_REGISTRY',
        'passed': admin_ok,
        'note': f'admin.allowed_tools={admin_agent.allowed_tools}',
    })

    # CHECK 6: audit log records tool calls with hospital/user attribution
    before = len(AUDIT_LOG)
    admin_agent.invoke_tool('search_expert_corpus', expert_id='chen', query='GGO protocol')
    checks.append({
        'label': 'audit_tool_call appends an attributed entry to AUDIT_LOG',
        'passed': len(AUDIT_LOG) == before + 1 and AUDIT_LOG[-1]['user_id'] == 'dr-admin',
        'note': f'AUDIT_LOG[-1]={AUDIT_LOG[-1]}',
    })

    # CHECK 7: epoch_invoke end-to-end, JWT -> role-bound tool call
    resp = epoch_invoke(f'Bearer {tok}', {'tool_name': 'search_expert_corpus', 'query': 'GGO protocol'})
    checks.append({
        'label': 'epoch_invoke performs JWT verification then a role-bound tool call',
        'passed': resp.get('status') == 200 and resp.get('hospital_id') == 'hospital_a',
        'note': f'status={resp.get("status")}, hospital_id={resp.get("hospital_id")}',
    })

    passed = sum(1 for c in checks if c['passed'])
    total = len(checks)
    return {'passed': passed == total, 'checks': checks, 'summary': f'{passed}/{total} checks passed'}


if __name__ == '__main__':
    print('\n╔══════════════════════════════════════════════════════╗')
    print('║  EPOCH — Session 15.2 Verification                   ║')
    print('╚══════════════════════════════════════════════════════╝\n')

    result = run_session_verification()
    print(f"  Verification: {result['summary']}\n")
    for check in result['checks']:
        icon = '✓' if check['passed'] else '✗'
        print(f"  {icon} {check['label']}")
        print(f"      {check['note']}")
    print()

    if not result['passed']:
        raise SystemExit(1)

    print('  ✓ Session 15.2 COMPLETE.\n')
