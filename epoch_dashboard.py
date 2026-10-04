"""
epoch_dashboard.py — Local test dashboard for the EPOCH capstone (15.1 + 15.2 + 15.3)

Not part of the graded sessions. A single FastAPI process that wires together
the real objects from all three modules so you can click through the three
isolation axes instead of reading verification output:

    DATA    (15.1) — FakeFAISSRouter scoped to (hospital_id, expert_id)
    ACTIONS (15.2) — JWT + ROLE_TOOLS RBAC, audited tool calls
    TIME    (15.3) — SLA-bounded agent runs, 206 partial content, circuit breaker

Run:
    source .venv/bin/activate
    python epoch_dashboard.py
    open http://127.0.0.1:8000

Everything here reuses the actual classes/functions from epoch_ingest.py,
epoch_auth.py, and epoch_sla.py — the dashboard does not reimplement any
isolation logic, it just gives you buttons to poke the real thing.
"""

import asyncio
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from epoch_ingest import router as faiss_router, rls_db, seed_corpus
from epoch_auth import (
    AUDIT_LOG, AuthError, ROLE_TOOLS, TOOL_REGISTRY,
    build_agent_for_role, mint_jwt, verify_jwt,
)
from epoch_sla import (
    CircuitBreaker, FakeCheckpointer, FakeEpochAgent,
    MIN_NOTES_FOR_206, SLA_BUDGET_SECONDS, epoch_invoke_sla,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared state — one process, one checkpointer, one breaker, one audit log.
# Restarting the dashboard resets all of it (in-memory, like every Fake* class
# in this capstone).
# ─────────────────────────────────────────────────────────────────────────────

seed_corpus()
CHECKPOINTER = FakeCheckpointer()
BREAKER = CircuitBreaker(window_size=20, failure_threshold=0.3, cooldown_seconds=15.0)

app = FastAPI(title='EPOCH Dashboard')


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class MintRequest(BaseModel):
    hospital_id: str = 'hospital_a'
    user_id: str = 'dr-jones'
    role: str = 'analyst'


class InvokeRequest(BaseModel):
    token: str
    expert_id: str = 'chen'
    question: str = 'What is your GGO follow-up protocol for a non-smoker?'
    scenario: str = 'complete'   # complete | partial | empty | custom
    custom_n: Optional[int] = None


class ToolCallRequest(BaseModel):
    token: str
    tool_name: str
    expert_id: str = 'chen'
    query: str = 'GGO protocol'
    patient_ref: str = 'patient-demo'
    plan: str = '3-month CT follow-up'


class SearchRequest(BaseModel):
    hospital_id: str = 'hospital_a'
    expert_id: str = 'chen'
    query: str = 'GGO protocol'


# ─────────────────────────────────────────────────────────────────────────────
# API — 15.1 data isolation
# ─────────────────────────────────────────────────────────────────────────────

@app.post('/api/search')
def api_search(req: SearchRequest):
    """Raw FAISS router search — bypasses auth on purpose, to demonstrate
    that isolation lives in the index keying, not in a permission check."""
    hits = faiss_router.search(req.hospital_id, req.expert_id, req.query)
    rls_rows = rls_db.select(req.hospital_id)
    return {'hospital_id': req.hospital_id, 'expert_id': req.expert_id,
            'hits': hits, 'rls_rows': rls_rows}


# ─────────────────────────────────────────────────────────────────────────────
# API — 15.2 action isolation
# ─────────────────────────────────────────────────────────────────────────────

@app.post('/api/mint')
def api_mint(req: MintRequest):
    token = mint_jwt(req.hospital_id, req.user_id, req.role)
    claims = verify_jwt(token)
    return {'token': token, 'claims': claims, 'allowed_tools': ROLE_TOOLS.get(req.role, [])}


@app.post('/api/tool-call')
def api_tool_call(req: ToolCallRequest):
    try:
        claims = verify_jwt(req.token)
    except AuthError as e:
        return {'status': 401, 'error': e.code, 'detail': e.detail}

    agent = build_agent_for_role(claims['role'], claims['hospital_id'], claims['sub'])
    try:
        result = agent.invoke_tool(
            req.tool_name,
            expert_id=req.expert_id, query=req.query,
            patient_ref=req.patient_ref, plan=req.plan,
        )
        return {'status': 200, 'role': claims['role'], 'result': result}
    except AuthError as e:
        return {'status': 403, 'error': e.code, 'detail': e.detail}
    except TypeError as e:
        return {'status': 400, 'error': 'bad_args', 'detail': str(e)}


@app.get('/api/audit')
def api_audit(limit: int = 25):
    entries = list(reversed(AUDIT_LOG[-limit:]))
    return {'count': len(AUDIT_LOG), 'entries': entries}


@app.get('/api/roles')
def api_roles():
    return {'role_tools': ROLE_TOOLS, 'tool_registry': list(TOOL_REGISTRY.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# API — 15.3 time isolation
# ─────────────────────────────────────────────────────────────────────────────

@app.post('/api/invoke')
async def api_invoke(req: InvokeRequest):
    if req.scenario == 'partial':
        agent = FakeEpochAgent(CHECKPOINTER, inject_failure_after_steps=3)
    elif req.scenario == 'empty':
        agent = FakeEpochAgent(CHECKPOINTER, inject_failure_after_steps=0)
    elif req.scenario == 'custom' and req.custom_n is not None:
        agent = FakeEpochAgent(CHECKPOINTER, inject_failure_after_steps=req.custom_n)
    else:
        agent = None  # runs to completion

    t0 = time.monotonic()
    resp = await epoch_invoke_sla(
        f'Bearer {req.token}',
        {'expert_id': req.expert_id, 'question': req.question},
        CHECKPOINTER, BREAKER, agent,
    )
    resp['elapsed_s'] = round(time.monotonic() - t0, 2)
    resp['sla_budget_s'] = SLA_BUDGET_SECONDS
    return resp


@app.get('/api/circuit')
def api_circuit():
    return {
        'state': BREAKER.state,
        'window': list(BREAKER.window),
        'failure_threshold': BREAKER.failure_threshold,
        'cooldown_seconds': BREAKER.cooldown_seconds,
        'opened_at': BREAKER.opened_at,
    }


@app.post('/api/circuit/trip')
def api_circuit_trip():
    """Force ten failures through the breaker — the 'downstream API is down
    for 8 minutes' war story, without waiting for real timeouts."""
    for _ in range(10):
        BREAKER.record(False)
    return api_circuit()


@app.post('/api/circuit/reset')
def api_circuit_reset():
    global BREAKER
    BREAKER = CircuitBreaker(window_size=20, failure_threshold=0.3, cooldown_seconds=15.0)
    return api_circuit()


@app.get('/api/health')
def api_health():
    return {'status': 'ok', 'circuit_state': BREAKER.state, 'audit_count': len(AUDIT_LOG)}


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard UI — single self-contained HTML page
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>EPOCH Dashboard</title>
<style>
  :root {
    --bg: #0b0f14; --panel: #131a22; --border: #22303c; --text: #e6edf3;
    --muted: #8b98a5; --accent: #4fa3ff; --green: #3fb950; --amber: #d29922;
    --red: #f85149; --purple: #a371f7;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); font-weight: 600;
       text-transform: uppercase; letter-spacing: 0.06em; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; min-width: 0;
  }
  .panel.wide { grid-column: 1 / -1; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }
  input, select, textarea {
    width: 100%; background: #0d1420; border: 1px solid var(--border); color: var(--text);
    padding: 7px 9px; border-radius: 6px; font-size: 13px; font-family: inherit;
  }
  textarea { resize: vertical; min-height: 50px; }
  button {
    background: var(--accent); color: #06121f; border: none; padding: 8px 14px;
    border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; margin-top: 12px;
    margin-right: 8px;
  }
  button.secondary { background: #26313d; color: var(--text); }
  button.danger { background: var(--red); color: #2b0b09; }
  button:hover { filter: brightness(1.1); }
  .row { display: flex; gap: 8px; }
  .row > * { flex: 1; }
  pre {
    background: #0d1420; border: 1px solid var(--border); border-radius: 6px;
    padding: 10px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
    max-height: 340px; overflow-y: auto; margin-top: 12px;
  }
  .badge {
    display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px;
    font-weight: 700; margin-left: 6px;
  }
  .b-200 { background: rgba(63,185,80,.15); color: var(--green); }
  .b-206 { background: rgba(210,153,34,.15); color: var(--amber); }
  .b-504, .b-503 { background: rgba(248,81,73,.15); color: var(--red); }
  .b-401, .b-403, .b-400 { background: rgba(163,113,247,.15); color: var(--purple); }
  .circuit-CLOSED { color: var(--green); }
  .circuit-OPEN { color: var(--red); }
  .circuit-HALF_OPEN { color: var(--amber); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
  th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  .muted { color: var(--muted); }
  .token-box { word-break: break-all; font-size: 11px; color: var(--muted); margin-top: 8px; }
  .dots span {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 3px;
  }
  .dot-ok { background: var(--green); } .dot-fail { background: var(--red); }
</style>
</head>
<body>

<h1>EPOCH Dashboard</h1>
<div class="subtitle">
  Local test surface for the capstone — DATA (15.1) · ACTIONS (15.2) · TIME (15.3).
  <span id="health" class="muted"></span>
</div>

<div class="grid">

  <!-- 15.2: Mint token -->
  <div class="panel">
    <h2>1 · Mint Identity (15.2)</h2>
    <label>Hospital</label>
    <select id="mint-hospital">
      <option value="hospital_a">hospital_a (Dr. Chen — pulmonology)</option>
      <option value="hospital_b">hospital_b (Dr. Patel — cardiology)</option>
    </select>
    <label>User ID</label>
    <input id="mint-user" value="dr-jones">
    <label>Role</label>
    <select id="mint-role">
      <option value="viewer">viewer (search only)</option>
      <option value="analyst" selected>analyst (search + patient lookup)</option>
      <option value="admin">admin (all tools)</option>
    </select>
    <button onclick="mintToken()">Mint JWT</button>
    <div class="token-box" id="token-display">No token minted yet.</div>
    <pre id="mint-out">—</pre>
  </div>

  <!-- 15.1: Data isolation -->
  <div class="panel">
    <h2>2 · Data Isolation (15.1)</h2>
    <label>Hospital</label>
    <select id="search-hospital">
      <option value="hospital_a">hospital_a</option>
      <option value="hospital_b">hospital_b</option>
    </select>
    <label>Expert corpus</label>
    <input id="search-expert" value="chen">
    <label>Query</label>
    <input id="search-query" value="GGO protocol">
    <button onclick="runSearch()">Search FAISS index</button>
    <button class="secondary" onclick="document.getElementById('search-hospital').value='hospital_b'; runSearch();">Try wrong hospital →</button>
    <pre id="search-out">Try hospital_a/chen (hits), then hospital_b/chen (nothing — wrong tenant).</pre>
  </div>

  <!-- 15.2: RBAC tool call -->
  <div class="panel">
    <h2>3 · RBAC Tool Call (15.2)</h2>
    <label>Tool</label>
    <select id="tool-name">
      <option value="search_expert_corpus">search_expert_corpus (all roles)</option>
      <option value="lookup_patient_history">lookup_patient_history (analyst+)</option>
      <option value="fetch_prior_imaging">fetch_prior_imaging (analyst+)</option>
      <option value="prescribe_followup">prescribe_followup (admin only)</option>
    </select>
    <div class="row">
      <div><label>patient_ref</label><input id="tool-patient" value="patient-demo"></div>
      <div><label>plan</label><input id="tool-plan" value="3-month CT follow-up"></div>
    </div>
    <button onclick="callTool()">Call tool with current token</button>
    <div class="muted" style="margin-top:6px; font-size:12px;">
      Try <b>prescribe_followup</b> as <b>viewer</b> or <b>analyst</b> → expect 403 (tool never bound, not just refused).
    </div>
    <pre id="tool-out">—</pre>
  </div>

  <!-- 15.3: SLA engine -->
  <div class="panel wide">
    <h2>4 · SLA Engine — Hard Deadline + HTTP 206 (15.3)</h2>
    <div class="row">
      <div>
        <label>Expert</label>
        <input id="inv-expert" value="chen">
      </div>
      <div>
        <label>Scenario</label>
        <select id="inv-scenario" onchange="toggleCustomN()">
          <option value="complete">Complete run (~4.5s, budget 5.0s → 200)</option>
          <option value="partial" selected>Cut cable after 3 steps → 206 partial</option>
          <option value="empty">Cut cable after 0 steps → 504 (deny-by-default)</option>
          <option value="custom">Custom fail-after-N steps</option>
        </select>
      </div>
      <div id="custom-n-wrap" style="display:none;">
        <label>Fail after N steps</label>
        <input id="inv-custom-n" type="number" value="1" min="0" max="7">
      </div>
    </div>
    <label>Question</label>
    <textarea id="inv-question">Dr. Chen, I have a 58-year-old non-smoking male with a 1.2cm ground-glass opacity, PET-negative. What is your diagnostic protocol?</textarea>
    <button onclick="invokeAgent()">Run agent (uses current token)</button>
    <div id="inv-status"></div>
    <pre id="inv-out">Mint a token first, then run a scenario.</pre>
  </div>

  <!-- Circuit breaker -->
  <div class="panel">
    <h2>5 · Circuit Breaker</h2>
    <p>State: <span id="circuit-state" class="circuit-CLOSED">CLOSED</span></p>
    <div class="dots" id="circuit-window"></div>
    <button class="danger" onclick="tripCircuit()">Force 10 failures → OPEN</button>
    <button class="secondary" onclick="resetCircuit()">Reset breaker</button>
    <div class="muted" style="margin-top:6px; font-size:12px;">
      After tripping OPEN, wait past the cooldown (15s) and run any agent scenario —
      the next call will be a single HALF_OPEN probe.
    </div>
    <pre id="circuit-out">—</pre>
  </div>

  <!-- Audit log -->
  <div class="panel">
    <h2>6 · Audit Log</h2>
    <button class="secondary" onclick="loadAudit()">Refresh</button>
    <table id="audit-table">
      <thead><tr><th>user</th><th>hospital</th><th>tool</th><th>time</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

</div>

<script>
let currentToken = null;

function badge(status) {
  return `<span class="badge b-${status}">${status}</span>`;
}

async function post(url, body) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
  return r.json();
}
async function get(url) {
  const r = await fetch(url);
  return r.json();
}

async function mintToken() {
  const body = {
    hospital_id: document.getElementById('mint-hospital').value,
    user_id: document.getElementById('mint-user').value,
    role: document.getElementById('mint-role').value,
  };
  const data = await post('/api/mint', body);
  currentToken = data.token;
  document.getElementById('token-display').innerText = 'token: ' + data.token.slice(0, 40) + '…';
  document.getElementById('mint-out').innerText = JSON.stringify(data, null, 2);
}

async function runSearch() {
  const body = {
    hospital_id: document.getElementById('search-hospital').value,
    expert_id: document.getElementById('search-expert').value,
    query: document.getElementById('search-query').value,
  };
  const data = await post('/api/search', body);
  document.getElementById('search-out').innerText = JSON.stringify(data, null, 2);
}

async function callTool() {
  if (!currentToken) { alert('Mint a token first.'); return; }
  const body = {
    token: currentToken,
    tool_name: document.getElementById('tool-name').value,
    patient_ref: document.getElementById('tool-patient').value,
    plan: document.getElementById('tool-plan').value,
    query: 'GGO protocol', expert_id: 'chen',
  };
  const data = await post('/api/tool-call', body);
  document.getElementById('tool-out').innerHTML = badge(data.status) + '\\n' + JSON.stringify(data, null, 2);
  loadAudit();
}

function toggleCustomN() {
  document.getElementById('custom-n-wrap').style.display =
    document.getElementById('inv-scenario').value === 'custom' ? 'block' : 'none';
}

async function invokeAgent() {
  if (!currentToken) { alert('Mint a token first.'); return; }
  const scenario = document.getElementById('inv-scenario').value;
  const body = {
    token: currentToken,
    expert_id: document.getElementById('inv-expert').value,
    question: document.getElementById('inv-question').value,
    scenario: scenario,
    custom_n: scenario === 'custom' ? parseInt(document.getElementById('inv-custom-n').value) : null,
  };
  document.getElementById('inv-status').innerHTML = 'running…';
  const data = await post('/api/invoke', body);
  document.getElementById('inv-status').innerHTML =
    badge(data.status) + ` <span class="muted">elapsed ${data.elapsed_s}s / budget ${data.sla_budget_s}s</span>`;
  document.getElementById('inv-out').innerText = JSON.stringify(data, null, 2);
  loadCircuit();
}

async function loadCircuit() {
  const data = await get('/api/circuit');
  const el = document.getElementById('circuit-state');
  el.innerText = data.state;
  el.className = 'circuit-' + data.state;
  document.getElementById('circuit-window').innerHTML =
    data.window.map(ok => `<span class="dot-${ok ? 'ok' : 'fail'}"></span>`).join('');
  document.getElementById('circuit-out').innerText = JSON.stringify(data, null, 2);
}
async function tripCircuit() {
  const data = await post('/api/circuit/trip');
  await loadCircuit();
}
async function resetCircuit() {
  const data = await post('/api/circuit/reset');
  await loadCircuit();
}

async function loadAudit() {
  const data = await get('/api/audit?limit=15');
  const tbody = document.querySelector('#audit-table tbody');
  tbody.innerHTML = data.entries.map(e => `
    <tr>
      <td>${e.user_id}</td>
      <td>${e.hospital_id}</td>
      <td>${e.tool_name}</td>
      <td class="muted">${new Date(e.ts * 1000).toLocaleTimeString()}</td>
    </tr>`).join('') || '<tr><td colspan="4" class="muted">No calls yet.</td></tr>';
}

async function loadHealth() {
  const data = await get('/api/health');
  document.getElementById('health').innerText =
    `· circuit: ${data.circuit_state} · audit entries: ${data.audit_count}`;
}

loadCircuit(); loadAudit(); loadHealth();
setInterval(loadHealth, 4000);
</script>
</body>
</html>
"""


@app.get('/', response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


if __name__ == '__main__':
    import uvicorn
    print('\n  EPOCH Dashboard → http://127.0.0.1:8000\n')
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='warning')
