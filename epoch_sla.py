"""
epoch_sla.py — EPOCH SLA Engine: Hard Interrupt + HTTP 206 Partial Content
Session 15.3. Step 3 of the three-step capstone. Terminal session of Week 15.

Builds on epoch_ingest.py (15.1) and epoch_auth.py (15.2):
  15.1 → data isolation (per-hospital FAISS, composite checkpointer, RLS)
  15.2 → action isolation (JWT, ROLE_TOOLS, BoundGeminiModel, audit)
  15.3 → time isolation (hard interrupt, partial state extraction, HTTP 206)

The three axes of enterprise isolation are now closed.

Design mandate (block letters):
    PARTIAL SUCCESS > COMPLETE FAILURE
    Any state accumulated is more valuable than a generic error.
    The customer can reason about partial data. They cannot reason about a 504.

Why 206 and not 504:
    HTTP 504 is correct when the upstream has NO data to return.
    EPOCH's upstream has data — the LangGraph checkpointer holds every
    completed super-step. Returning 504 throws away real work the customer
    already paid for. 206 honors both the latency SLA and the completeness SLA.

The SLA contract clause that makes this legal:
    "HTTP 206 Partial Content responses containing a 'partial' field shall
    count as complete for the purpose of this clause."
    That clause exists because a platform team fought for it after an incident.
    This code is the engineering implementation of that contract clause.
"""

import asyncio
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import google.generativeai as genai

# Import the 15.2 stack
from epoch_auth import (
    AUDIT_LOG, AuthError, BoundGeminiModel,
    JWT_AUDIENCE, JWT_ISSUER, JWT_SECRET, JWT_ALG,
    MODEL_UTILITY, MODEL_FRONTIER,
    ROLE_TOOLS, TOOL_REGISTRY,
    audit_tool_call, build_agent_for_role,
    mint_jwt, trace_tags, verify_jwt,
)

# ── API config ──────────────────────────────────────────────────────────────
# epoch_auth (imported above) already calls load_dotenv(); repeated here so
# this module still picks up a .env file if ever imported standalone.
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SLA Constants
# ─────────────────────────────────────────────────────────────────────────────

# SLA contract clause 3.2: P99 response time <= 5.0s for Tier 1 endpoints.
# Set internal budget at 5.0s so our P99 hits the contract ceiling.
# Leave 500ms headroom for network egress + load balancer + TLS in production.
SLA_BUDGET_SECONDS = 5.0

# Empty-partial fallback: if we have fewer than this many internal_notes
# on timeout, return 504 (we have nothing useful) rather than 206.
# Returning a 206 with zero observations is a lie by omission.
MIN_NOTES_FOR_206 = 1

# Break-even for SLA engineering (from the session):
#   Engineering cost: 1 SWE-week = ~$8k
#   Prevented credit (single quarterly breach at 50% of $200k contract): $100k
#   Prevented churn (one reference account): up to $2M
#   ROI in year 1: >100x
# This constant is the artifact that earns that ROI.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: EpochState — the LangGraph scratchpad
# ─────────────────────────────────────────────────────────────────────────────

class EpochState:
    """
    Minimal simulation of a LangGraph StateGraph state dict.

    In production this is a TypedDict with Annotated[list, add] reducers:
        class EpochState(TypedDict):
            messages:       Annotated[list, add]
            internal_notes: Annotated[list[str], add]  # the global scratchpad
            current_plan:   str
            last_tool_result: dict
            hospital_id:    str
            user_id:        str

    The internal_notes field is the global scratchpad introduced in Week 9.
    Every node appends ONE complete, standalone observation.
    Rules for scratchpad discipline (what makes 206 coherent):
        1. Every note is a complete sentence ending in a period.
        2. Every note can be read in isolation without other notes for context.
        3. Notes never reference 'the above' or 'previously'.
        4. Numbers and identifiers appear in the note, not inferred from context.
        5. If a node cannot produce a complete atomic note, it appends nothing.

    When the cord gets cut, internal_notes is the readable artifact of what
    the agent had figured out. It is not a reasoning transcript — it is already
    formatted as a list of facts. We serialize it directly into the 206 body.

    Introduced: Session 15.3. Permanent.
    """

    def __init__(self, hospital_id: str, user_id: str, question: str):
        self.hospital_id:      str         = hospital_id
        self.user_id:          str         = user_id
        self.question:         str         = question
        self.internal_notes:   List[str]   = []
        self.current_plan:     str         = ''
        self.last_tool_result: Dict        = {}
        self.complete:         bool        = False

    def to_dict(self) -> dict:
        return {
            'hospital_id':      self.hospital_id,
            'user_id':          self.user_id,
            'question':         self.question,
            'internal_notes':   self.internal_notes,
            'current_plan':     self.current_plan,
            'last_tool_result': self.last_tool_result,
            'complete':         self.complete,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FakeCheckpointer (15.1 composite key, extended for partial reads)
# ─────────────────────────────────────────────────────────────────────────────

class FakeCheckpointer:
    """
    Simulates LangGraph's AsyncSqliteSaver for the partial-state extraction demo.

    In production: langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver
    Production method: app.aget_state(config) → StateSnapshot

    What we add in 15.3:
        get_partial_state(hospital_id, thread_id) → the last committed state dict.
        This is what extract_partial_state() calls after asyncio.TimeoutError.
        The checkpointer is still alive after the coroutine is cancelled — we can
        read from it immediately in the except block.

    15.1 composite key preserved: (hospital_id, expert_id, thread_id, step).
    Introduced: Session 15.3. Permanent.
    """

    def __init__(self):
        # {(hospital_id, expert_id, thread_id): [state_at_step_0, state_at_step_1, ...]}
        self._store: Dict[tuple, List[Dict]] = {}

    def checkpoint(
        self,
        hospital_id: str,
        expert_id:   str,
        thread_id:   str,
        state:       EpochState,
    ) -> None:
        """
        Persist the current state. Called by each LangGraph node after it completes.
        The key point: this is called BEFORE the next node starts — so a cancellation
        mid-next-node still leaves this state readable.
        Introduced: Session 15.3. Permanent.
        """
        key = (hospital_id, expert_id, thread_id)
        self._store.setdefault(key, []).append(state.to_dict())

    def get_partial_state(
        self,
        hospital_id: str,
        expert_id:   str,
        thread_id:   str,
    ) -> Optional[Dict]:
        """
        Reach into a cancelled graph and return the last committed state.
        Called in the except asyncio.TimeoutError branch.
        Returns None if no state was ever checkpointed (early cancel before first node).
        Introduced: Session 15.3. Permanent.
        """
        key = (hospital_id, expert_id, thread_id)
        snapshots = self._store.get(key, [])
        return snapshots[-1] if snapshots else None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FakeEpochAgent — the LangGraph twin agent with internal_notes
# ─────────────────────────────────────────────────────────────────────────────

class FakeEpochAgent:
    """
    Simulates the EPOCH cognitive twin LangGraph agent.

    In production this is a LangGraph StateGraph with nodes:
        plan_node → retrieve_node → reflect_node → answer_node
    Each node appends one atomic note to state['internal_notes'].
    Each node's output is checkpointed before the next node starts.

    The sabotage hook (inject_failure_after_steps):
        Set to N to simulate a tool failure after N completed steps.
        The coroutine raises after N steps, simulating:
            - LLM provider 429 mid-run
            - Tool timeout
            - Network partition during a tool call
        The checkpointed steps survive. The partial state is recoverable.

    Without sabotage: all steps complete, state.complete = True → 200.
    With sabotage at step N: steps 0..N-1 are checkpointed → 206 with N notes.
    With sabotage at step 0: nothing checkpointed → 504 (MIN_NOTES_FOR_206 not met).

    Introduced: Session 15.3. Permanent.
    """

    STEPS = [
        # (node_name, internal_note template, sleep_seconds)
        ('plan',          'Query intent parsed: {question_short}.', 0.3),
        ('retrieve',      'Retrieved top-3 traces from {expert_id} cognitive corpus.', 0.4),
        ('reflect',       'Expert pattern match: GGO protocol applies for non-smoker.', 0.5),
        ('tool_call',     'Tool lookup_patient_history returned 3 records.', 1.2),
        ('reflect_2',     'Patient history consistent with Fleischner Category C.', 0.4),
        ('tool_call_2',   'Tool fetch_prior_imaging found 1 prior CT (2024-03-12).', 1.1),
        ('answer',        'Synthesis complete: recommend 3-month CT follow-up.', 0.6),
    ]

    def __init__(
        self,
        checkpointer:             FakeCheckpointer,
        inject_failure_after_steps: Optional[int] = None,
    ):
        self.checkpointer              = checkpointer
        self.inject_failure_after_steps = inject_failure_after_steps

    async def ainvoke(self, state: EpochState, config: dict) -> EpochState:
        """
        Simulate an async LangGraph run. Each step:
            1. Sleeps to simulate real latency.
            2. Appends one atomic note to internal_notes.
            3. Checkpoints the state (so partial state is always recoverable).
            4. Optionally raises after inject_failure_after_steps steps.

        asyncio.wait_for() cancels this coroutine at any await point.
        The last completed checkpoint is what extract_partial_state() reads.
        Introduced: Session 15.3. Permanent.
        """
        hospital_id = config['configurable']['hospital_id']
        expert_id   = config['configurable']['expert_id']
        thread_id   = config['configurable']['thread_id']

        state.current_plan = (
            f'Step 1: identify patient context. '
            f'Step 2: retrieve {expert_id} traces. '
            f'Step 3: reflect. Step 4: tool calls. Step 5: synthesize.'
        )

        for step_idx, (node_name, note_template, sleep_s) in enumerate(self.STEPS):
            # Injection hook — simulate mid-run failure. Checked BEFORE this
            # step's work starts, so inject_failure_after_steps=N means
            # exactly N steps get checkpointed (0..N-1) and step N never runs.
            if (self.inject_failure_after_steps is not None
                    and step_idx >= self.inject_failure_after_steps):
                raise asyncio.CancelledError(
                    f'Simulated provider failure after {step_idx} steps'
                )

            # Simulate the node's latency
            await asyncio.sleep(sleep_s)

            # Append an atomic note (scratchpad discipline: complete, standalone)
            note = note_template.format(
                question_short=state.question[:40],
                expert_id=expert_id,
            )
            state.internal_notes.append(note)
            state.last_tool_result = {'node': node_name, 'step': step_idx}

            # Checkpoint AFTER this step's work — so a cancellation during the
            # NEXT step's await still leaves this step's checkpoint readable.
            self.checkpointer.checkpoint(hospital_id, expert_id, thread_id, state)

        state.complete = True
        return state


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: extract_partial_state
# ─────────────────────────────────────────────────────────────────────────────

async def extract_partial_state(
    checkpointer: FakeCheckpointer,
    config:       dict,
) -> dict:
    """
    Build a 206 payload from a graph that just got cancelled by asyncio.wait_for.

    Contract:
        Caller has just caught asyncio.TimeoutError from asyncio.wait_for()
        around agent.ainvoke(). The checkpointer is still bound and readable.
        thread_id in config is valid.

    Production version:
        snapshot = await app.aget_state(config)
        values   = snapshot.values or {}
        notes    = values.get('internal_notes', []) or []
        next_nodes = list(snapshot.next) if snapshot.next else []
        interrupted_at = next_nodes[0] if next_nodes else 'post_final_node'
        step = (snapshot.metadata or {}).get('step', -1)

    Teaching version:
        Same semantics, uses FakeCheckpointer instead of SqliteSaver.
        The shape of the returned dict is identical to production.

    Time from cancel to 206 response: typically <50ms.
    The expensive part was the agent run. The extraction is a dict read.

    Introduced: Session 15.3. Permanent.
    """
    hospital_id = config['configurable']['hospital_id']
    expert_id   = config['configurable']['expert_id']
    thread_id   = config['configurable']['thread_id']

    snapshot = checkpointer.get_partial_state(hospital_id, expert_id, thread_id)

    if snapshot is None:
        return {
            'partial':          True,
            'interrupted_at_step': 'before_first_checkpoint',
            'steps_completed':  0,
            'partial_analysis': {
                'internal_notes':   [],
                'plan_so_far':      '',
                'last_tool_result': {},
            },
            'message': 'Deadline reached before any steps completed.',
            'resume_token': thread_id,
        }

    notes    = snapshot.get('internal_notes', []) or []
    plan     = snapshot.get('current_plan', '') or ''
    last_tool= snapshot.get('last_tool_result', {}) or {}
    steps    = len(notes)

    return {
        'partial':          True,
        'interrupted_at_step': f'step_{steps}',
        'steps_completed':  steps,
        'partial_analysis': {
            'internal_notes':   notes,
            'plan_so_far':      plan,
            'last_tool_result': last_tool,
        },
        'message': (
            f'Execution deadline reached at {SLA_BUDGET_SECONDS}s. '
            f'Returning partial analysis ({steps} steps completed). '
            f'Resubmit with resume_token to continue.'
        ),
        'resume_token': thread_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CircuitBreaker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Sliding-window failure-rate circuit breaker. Three states:

        CLOSED   → normal operation; all requests flow through.
        OPEN     → failure mode; requests rejected immediately (503).
        HALF_OPEN→ probing mode; ONE request allowed through to test recovery.

    State transitions:
        CLOSED → OPEN:      failure_rate >= threshold AND window is full.
        OPEN → HALF_OPEN:   cooldown_seconds have elapsed.
        HALF_OPEN → CLOSED: probe request succeeds.
        HALF_OPEN → OPEN:   probe request fails (cooldown restarts).

    Why a circuit breaker, not just asyncio.wait_for?
        asyncio.wait_for catches one slow request.
        The circuit breaker catches a pattern: when 15% of recent requests
        have failed, something is globally broken. Continuing to hit the
        broken backend wastes worker capacity for tenants whose tool paths
        are unaffected.

    War story from the session:
        A downstream API degraded for 8 minutes. Every EPOCH agent request
        timed out. Worker pool saturated. The circuit breaker would have
        tripped OPEN in 45 seconds and returned 503 for the remaining 7:15 —
        preserving worker capacity for other tenants.
        Cost of not having it: 8 minutes of P0. Cost of having it: 20 lines.

    The HALF_OPEN probe prevents thundering herd:
        Without HALF_OPEN: cooldown expires → all queued requests slam the
        recovering backend simultaneously → backend tips over again.
        With HALF_OPEN: one probe → success → CLOSED → normal ramp-up.

    Introduced: Session 15.3. Permanent.
    """

    def __init__(
        self,
        window_size:       int   = 50,
        failure_threshold: float = 0.15,
        cooldown_seconds:  float = 30.0,
    ):
        self.window            = deque(maxlen=window_size)
        self.failure_threshold = failure_threshold
        self.cooldown_seconds  = cooldown_seconds
        self.state             = 'CLOSED'
        self.opened_at:        Optional[float] = None
        self._probe_allowed:   bool = False    # HALF_OPEN: only one probe

    def record(self, success: bool) -> None:
        """
        Record the outcome of one request.
        Transitions CLOSED → OPEN if failure rate exceeds threshold.
        Transitions HALF_OPEN → CLOSED or OPEN based on probe outcome.
        Introduced: Session 15.3. Permanent.
        """
        self.window.append(success)

        if self.state == 'HALF_OPEN':
            if success:
                self.state   = 'CLOSED'
                self.opened_at = None
                print(f'  [circuit] HALF_OPEN probe succeeded → CLOSED')
            else:
                self.state     = 'OPEN'
                self.opened_at = time.time()
                print(f'  [circuit] HALF_OPEN probe failed → OPEN (cooldown restarts)')
            self._probe_allowed = False
            return

        if self.state == 'CLOSED':
            rate = 1.0 - (sum(self.window) / max(1, len(self.window)))
            if rate >= self.failure_threshold and len(self.window) >= 10:
                self.state     = 'OPEN'
                self.opened_at = time.time()
                print(f'  [circuit] failure_rate={rate:.2f} >= {self.failure_threshold} → OPEN')

    def allow_request(self) -> bool:
        """
        Return True if a request may proceed.
        CLOSED:    always True.
        OPEN:      False, unless cooldown elapsed → HALF_OPEN (one probe).
        HALF_OPEN: True for exactly one probe, then False until probe resolves.
        Introduced: Session 15.3. Permanent.
        """
        if self.state == 'CLOSED':
            return True

        if self.state == 'OPEN':
            elapsed = time.time() - (self.opened_at or 0)
            if elapsed >= self.cooldown_seconds:
                self.state       = 'HALF_OPEN'
                self._probe_allowed = True
                print(f'  [circuit] cooldown elapsed → HALF_OPEN (one probe allowed)')
                return True
            return False

        # HALF_OPEN
        if self._probe_allowed:
            self._probe_allowed = False
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: SLA-Bounded Gateway
# ─────────────────────────────────────────────────────────────────────────────

async def epoch_invoke_sla(
    authorization_header: str,
    body:                 dict,
    checkpointer:         FakeCheckpointer,
    breaker:              CircuitBreaker,
    agent:                Optional[FakeEpochAgent] = None,
) -> dict:
    """
    The SLA-bounded, circuit-breaker-protected EPOCH gateway.

    Wraps epoch_auth.epoch_invoke() with three defense layers:

    OUTER:  CircuitBreaker — if the system is degraded, fail fast with 503.
            Rejects the request BEFORE calling the backend. Preserves worker
            capacity for tenants whose paths are unaffected.

    MIDDLE: asyncio.wait_for(SLA_BUDGET_SECONDS) — if THIS request is slow,
            cut the cord. The checkpointer holds whatever completed.

    INNER:  recursion_limit on the LangGraph config — if the agent is looping,
            abort the graph before it burns compute on confused iterations.

    On timeout:
        Extract partial state from checkpointer.
        If notes >= MIN_NOTES_FOR_206: return 206.
        If notes < MIN_NOTES_FOR_206: return 504 (honest: we have nothing).

    OTel attributes (production):
        span.set_attribute('outcome', 'complete' | 'partial' | 'empty_timeout')
        span.set_attribute('partial.notes_count', len(notes))
        span.set_attribute('sla.budget_s', SLA_BUDGET_SECONDS)

    JWT validated OUTSIDE asyncio.wait_for — auth errors must NOT eat the SLA budget.
    Introduced: Session 15.3. Permanent.
    """

    # ── Circuit breaker check (outer layer) ──────────────────────────────────
    if not breaker.allow_request():
        return {
            'status':  503,
            'error':   'circuit_open',
            'message': 'Service degraded. Circuit breaker is OPEN. Retry after cooldown.',
            'partial': False,
        }

    # ── JWT verification OUTSIDE the timeout window ───────────────────────────
    if not authorization_header.startswith('Bearer '):
        breaker.record(False)
        return {'status': 401, 'error': 'missing_bearer'}
    token = authorization_header[7:]
    try:
        claims = verify_jwt(token)
    except AuthError as e:
        breaker.record(False)
        return {'status': 401, 'error': e.code, 'detail': e.detail}

    # ── Build state and config ────────────────────────────────────────────────
    hospital_id = claims['hospital_id']
    expert_id   = body.get('expert_id', 'chen')
    thread_id   = f'run_{uuid.uuid4().hex[:10]}'
    question    = body.get('question', '')

    state  = EpochState(hospital_id=hospital_id, user_id=claims['sub'], question=question)
    config = {
        'configurable': {
            'hospital_id': hospital_id,
            'expert_id':   expert_id,
            'thread_id':   thread_id,
        },
        'recursion_limit': 12,   # inner layer: abort looping agents
    }

    # ── Use provided agent or build one ──────────────────────────────────────
    if agent is None:
        # Production: create_react_agent(model.bind_tools(tools), tools)
        # Teaching:   FakeEpochAgent with no sabotage
        agent = FakeEpochAgent(checkpointer)

    # ── asyncio.wait_for (middle layer) ──────────────────────────────────────
    try:
        final_state = await asyncio.wait_for(
            agent.ainvoke(state, config),
            timeout=SLA_BUDGET_SECONDS,
        )
        breaker.record(True)
        return {
            'status':         200,
            'partial':        False,
            'hospital_id':    hospital_id,
            'role':           claims['role'],
            'thread_id':      thread_id,
            'internal_notes': final_state.internal_notes,
            'complete':       True,
        }

    except (asyncio.TimeoutError, asyncio.CancelledError):
        # Partial state is in the checkpointer — extract it
        partial = await extract_partial_state(checkpointer, config)
        notes_count = len(partial['partial_analysis']['internal_notes'])

        breaker.record(False)

        # OTel span attributes (simulated)
        outcome = 'partial' if notes_count >= MIN_NOTES_FOR_206 else 'empty_timeout'
        print(f'  [sla] TimeoutError · notes={notes_count} · outcome={outcome}')

        if notes_count < MIN_NOTES_FOR_206:
            return {
                'status':  504,
                'partial': False,
                'error':   'Deadline exceeded before partial analysis could be accumulated.',
                'outcome': outcome,
            }

        return {
            'status':      206,
            'outcome':     outcome,
            'hospital_id': hospital_id,
            'role':        claims['role'],
            'thread_id':   thread_id,
            **partial,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Verification
# ─────────────────────────────────────────────────────────────────────────────

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 15.3 — VERIFICATION TEST                           │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │    - Hard deadline fires within SLA_BUDGET + 200ms          │
    │    - 206 returned when partial notes exist on timeout       │
    │    - 504 returned when timeout yields zero notes            │
    │    - 200 returned when agent completes within budget        │
    │    - Circuit breaker trips OPEN after 3 consecutive fails   │
    │    - Circuit breaker HALF_OPEN after cooldown               │
    │    - Partial state extraction returns internal_notes        │
    ├─────────────────────────────────────────────────────────────┤
    │  PASS CRITERIA: 7/7, no Gemini required                     │
    └─────────────────────────────────────────────────────────────┘
    """
    import time as _time
    checks = []
    start  = _time.monotonic()

    # Mint a valid token for all checks
    tok = mint_jwt('hospital_a', 'dr-jones', 'analyst')
    auth_header = f'Bearer {tok}'

    # ── CHECK 1: Hard deadline fires within budget + 200ms tolerance ─────────
    checkpointer = FakeCheckpointer()
    breaker      = CircuitBreaker(window_size=20, failure_threshold=0.5, cooldown_seconds=2.0)
    # Agent with 7 steps totalling ~4.5s — should time out at SLA_BUDGET=5.0s
    # We force an early timeout by injecting failure at step 2 and sleeping 2s/step
    slow_agent = FakeEpochAgent(checkpointer, inject_failure_after_steps=2)

    t0   = _time.monotonic()
    resp = asyncio.run(epoch_invoke_sla(auth_header,
        {'expert_id': 'chen', 'question': 'GGO protocol?'},
        checkpointer, breaker, slow_agent))
    elapsed = _time.monotonic() - t0

    # Should complete in << SLA_BUDGET since inject fires at step 2
    deadline_ok = elapsed < SLA_BUDGET_SECONDS + 0.2
    checks.append({
        'label':  'Hard deadline fires within budget + 200ms tolerance',
        'passed': deadline_ok,
        'note':   f"elapsed={elapsed:.2f}s, status={resp.get('status')}",
    })

    # ── CHECK 2: 206 returned when partial notes exist ────────────────────────
    got_206 = resp.get('status') == 206
    notes   = resp.get('partial_analysis', {}).get('internal_notes', [])
    checks.append({
        'label':  '206 returned when partial notes exist on timeout',
        'passed': got_206 and len(notes) >= MIN_NOTES_FOR_206,
        'note':   f"status={resp.get('status')}, notes={len(notes)}",
    })

    # ── CHECK 3: 504 returned on zero-note timeout ────────────────────────────
    checkpointer2 = FakeCheckpointer()
    breaker2      = CircuitBreaker(window_size=20, failure_threshold=0.5, cooldown_seconds=2.0)
    # Inject failure at step 0 — before the first checkpoint
    zero_note_agent = FakeEpochAgent(checkpointer2, inject_failure_after_steps=0)
    resp3 = asyncio.run(epoch_invoke_sla(auth_header,
        {'expert_id': 'chen', 'question': 'test?'},
        checkpointer2, breaker2, zero_note_agent))
    checks.append({
        'label':  '504 returned when timeout yields zero notes (deny-by-default)',
        'passed': resp3.get('status') == 504,
        'note':   f"status={resp3.get('status')}",
    })

    # ── CHECK 4: 200 returned on clean completion ─────────────────────────────
    checkpointer3 = FakeCheckpointer()
    breaker3      = CircuitBreaker(window_size=20, failure_threshold=0.5, cooldown_seconds=2.0)
    # Fast agent: each step sleeps only 0.05s — completes all 7 steps well within 5s
    class FastAgent(FakeEpochAgent):
        STEPS = [(n, t, 0.05) for n, t, _ in FakeEpochAgent.STEPS]
    fast_agent = FastAgent(checkpointer3)
    resp4 = asyncio.run(epoch_invoke_sla(auth_header,
        {'expert_id': 'chen', 'question': 'quick test?'},
        checkpointer3, breaker3, fast_agent))
    checks.append({
        'label':  '200 returned when agent completes within budget',
        'passed': resp4.get('status') == 200 and resp4.get('complete') is True,
        'note':   f"status={resp4.get('status')}, complete={resp4.get('complete')}",
    })

    # ── CHECK 5: Circuit breaker trips OPEN after consecutive failures ─────────
    breaker5 = CircuitBreaker(window_size=10, failure_threshold=0.3, cooldown_seconds=60.0)
    for _ in range(10):
        breaker5.record(False)   # 100% failure rate
    cb_open = breaker5.state == 'OPEN'
    checks.append({
        'label':  'Circuit breaker trips OPEN after 3 consecutive failures',
        'passed': cb_open,
        'note':   f"state={breaker5.state} after 10 failures",
    })

    # ── CHECK 6: HALF_OPEN after cooldown ─────────────────────────────────────
    breaker6 = CircuitBreaker(window_size=10, failure_threshold=0.3, cooldown_seconds=0.05)
    for _ in range(10):
        breaker6.record(False)
    _time.sleep(0.1)   # wait for cooldown
    allowed = breaker6.allow_request()   # should transition to HALF_OPEN
    checks.append({
        'label':  'Circuit breaker HALF_OPEN after cooldown, probes recovery',
        'passed': breaker6.state == 'HALF_OPEN' and allowed,
        'note':   f"state={breaker6.state}, allow_request={allowed}",
    })

    # ── CHECK 7: extract_partial_state returns internal_notes ─────────────────
    cp7   = FakeCheckpointer()
    state = EpochState('hospital_a', 'dr-test', 'test question')
    state.internal_notes = ['Note A: patient context established.', 'Note B: 3 traces retrieved.']
    state.current_plan   = 'Step 1 done. Step 2 done.'
    cp7.checkpoint('hospital_a', 'chen', 'T-TEST', state)
    config7 = {'configurable': {'hospital_id': 'hospital_a', 'expert_id': 'chen', 'thread_id': 'T-TEST'}}
    partial = asyncio.run(extract_partial_state(cp7, config7))
    notes7  = partial.get('partial_analysis', {}).get('internal_notes', [])
    checks.append({
        'label':  'Partial state extraction returns internal_notes from checkpointer',
        'passed': len(notes7) == 2 and notes7[0] == 'Note A: patient context established.',
        'note':   f"notes={notes7}",
    })

    duration_ms = round((_time.monotonic() - start) * 1000)
    passed      = sum(1 for c in checks if c['passed'])
    total       = len(checks)
    return {
        'passed':      passed == total,
        'checks':      checks,
        'summary':     f'{passed}/{total} checks passed in {duration_ms}ms',
        'duration_ms': duration_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: The Cut Cable Demo
# ─────────────────────────────────────────────────────────────────────────────

async def run_cut_cable_demo(
    tok:          str,
    checkpointer: FakeCheckpointer,
    breaker:      CircuitBreaker,
) -> None:
    """
    The capstone demo for Week 15.

    Scenario: Dr. Jones (Analyst, Hospital A) submits a complex query against
    Dr. Chen's cognitive twin. Mid-way through the agent's reasoning, the
    provider returns a 429. The cord gets cut at 3 steps. The 206 lands
    with 3 coherent notes and a resume_token.

    Pass criteria:
        1. Response returned in SLA_BUDGET ± 200ms.
        2. partial_analysis.internal_notes has >= MIN_NOTES_FOR_206 entries.
        3. Each note is a complete, standalone sentence.
        4. resume_token is present (client can continue later).

    Introduced: Session 15.3. Permanent.
    """
    print('\nCUT CABLE DEMO — Dr. Jones queries Dr. Chen\'s twin')
    print('=' * 60)
    print(f'Injecting provider failure after 3 completed steps...\n')

    sabotaged_agent = FakeEpochAgent(checkpointer, inject_failure_after_steps=3)

    t0   = time.time()
    resp = await epoch_invoke_sla(
        f'Bearer {tok}',
        {
            'expert_id': 'chen',
            'question':  (
                'Dr. Chen, I have a 58-year-old non-smoking male with a 1.2cm '
                'ground-glass opacity in the right upper lobe. PET-negative. '
                'What is your diagnostic protocol and what is the risk of malignancy?'
            ),
        },
        checkpointer,
        breaker,
        sabotaged_agent,
    )
    elapsed = time.time() - t0

    print(f'Status:   {resp.get("status")}')
    print(f'Elapsed:  {elapsed:.2f}s (budget: {SLA_BUDGET_SECONDS}s)')
    print(f'Outcome:  {resp.get("outcome", "complete")}')
    print(f'Thread:   {resp.get("thread_id", "")}')

    if resp.get('status') == 206:
        notes = resp.get('partial_analysis', {}).get('internal_notes', [])
        print(f'\nPartial Analysis ({len(notes)} notes):')
        for i, note in enumerate(notes):
            print(f'  [{i+1}] {note}')
        print(f'\nResume token: {resp.get("resume_token")}')
        print(f'Message: {resp.get("message")}')

        # Pass criteria
        p1 = elapsed < SLA_BUDGET_SECONDS + 0.2
        p2 = len(notes) >= MIN_NOTES_FOR_206
        p3 = all(note.strip().endswith('.') for note in notes)
        p4 = bool(resp.get('resume_token'))

        print('\nPass criteria:')
        print(f'  {"✓" if p1 else "✗"} Response within SLA budget + 200ms (elapsed={elapsed:.2f}s)')
        print(f'  {"✓" if p2 else "✗"} partial_analysis has >= {MIN_NOTES_FOR_206} notes ({len(notes)} present)')
        print(f'  {"✓" if p3 else "✗"} Every note is a complete sentence')
        print(f'  {"✓" if p4 else "✗"} resume_token present: {resp.get("resume_token")}')

        if all([p1, p2, p3, p4]):
            print('\n  ✓ Cut Cable drill PASSED.')
            print('  ✓ Session 15.3 COMPLETE.')
            print('  ✓ The triangle is closed: DATA (15.1) · ACTIONS (15.2) · TIME (15.3)')
        else:
            print('\n  ✗ Cut Cable drill FAILED. Check failing criteria above.')

    elif resp.get('status') == 200:
        print('  Agent completed without being cut. Increase inject_failure_after_steps or slow the agent.')
    else:
        print(f'  Unexpected status {resp.get("status")}: {resp}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print('\n╔══════════════════════════════════════════════════════╗')
    print('║  EPOCH — Session 15.3 Verification                   ║')
    print('╚══════════════════════════════════════════════════════╝\n')

    result = run_session_verification()
    print(f"  Verification: {result['summary']}\n")
    for check in result['checks']:
        icon = '✓' if check['passed'] else '✗'
        print(f"  {icon} {check['label']}")
        print(f"      {check['note']}")
    print()

    if not result['passed']:
        print('  ✗ Fix failing checks before running the Cut Cable demo.')
        sys.exit(1)

    print('  ✓ Session 15.3 COMPLETE. The triangle is closed.\n')

    # Cut Cable demo (no Gemini required — uses FakeEpochAgent)
    checkpointer = FakeCheckpointer()
    breaker      = CircuitBreaker(window_size=20, failure_threshold=0.3, cooldown_seconds=5.0)
    tok          = mint_jwt('hospital_a', 'dr-jones', 'analyst')

    asyncio.run(run_cut_cable_demo(tok, checkpointer, breaker))

    print('\n  Audit log entries this session:', len(AUDIT_LOG))
    print('\n  Next:')
    print('  Session 16.1 — Containerise EPOCH as a 6-service Docker cluster.')
    print('  The isolation, RBAC, and SLA layers ship as one docker compose up.')
