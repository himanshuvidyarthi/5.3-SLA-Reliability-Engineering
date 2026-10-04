# EPOCH — Week 15 Capstone

EPOCH is a teaching build of an enterprise AI gateway for hospital "cognitive
twin" agents (e.g. a LangGraph agent trained on Dr. Chen's diagnostic
reasoning). Across three sessions it closes three axes of multi-tenant
isolation. Everything runs locally with in-memory `Fake*` stand-ins for real
infrastructure (FAISS, Postgres, LangGraph's checkpointer) — no external
services or API keys are required to run or validate it.

```
   15.1  DATA     isolation  →  a hospital can never read another hospital's vectors
   15.2  ACTIONS  isolation  →  a role can never call a tool it wasn't bound to
   15.3  TIME     isolation  →  a slow agent run degrades to 206, never hangs
```

---

## 1. Idiot-proof setup (clone → running dashboard in under 2 minutes)

**Prerequisites:** Python 3.11+ and `git`. Nothing else — no Docker, no
database, no API keys.

```bash
# 1. Clone the repo
git clone <this-repo-url> epoch
cd epoch

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) copy the env template — see "Environment variables" below
cp .env.example .env

# 5. Run the three session verifications (each should print "N/N checks passed")
python epoch_ingest.py           # Session 15.1
python epoch_auth.py             # Session 15.2
python epoch_sla.py              # Session 15.3 + the Cut Cable demo

# 6. Launch the local dashboard
python epoch_dashboard.py
# then open http://127.0.0.1:8000 in a browser
```

That's it — steps 1–6 are the entire setup, and step 4 is skippable. No
`.env` file and no `GEMINI_API_KEY` are needed: every check and every
dashboard action runs against the `Fake*` simulators, never a live Gemini
call.

### Environment variables

`.env.example` documents the two variables the code will read if present
(via `python-dotenv`, loaded automatically at import time — no manual
`export` needed once you have a `.env`):

| Variable | Required? | Default if unset | Used for |
|---|---|---|---|
| `GEMINI_API_KEY` | No | `''` | Only consulted if you swap `FakeEpochAgent` for a real Gemini-backed agent. Nothing in the current code path calls the live API. |
| `JWT_SECRET` | No | a hardcoded teaching placeholder in `epoch_auth.py` | The HMAC key used to sign/verify JWTs. The placeholder is fine for local demos; set a real one before any real deployment, since anyone reading the source knows the default. |

`.env` is git-ignored, so it's safe to put real values in it locally.

**Troubleshooting**
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` on startup | You forgot to `source .venv/bin/activate`, or `pip install -r requirements.txt` didn't run in that venv. |
| `Address already in use` on port 8000 | Another dashboard instance is running — find it with `lsof -i :8000` and `kill <pid>`, or stop your previous session. |
| Verification prints `✗` anywhere | Something in this checkout was edited — re-clone or `git diff` against the last known-good commit. |

---

## 2. What was built, step by step

### Step 0 — Environment
An empty directory with only a `.venv`. No prior code existed, so sessions
15.1 and 15.2 (normally prerequisites) were built first to give 15.3
something real to import, using the same design mandates the 15.3 spec
describes (deny-by-default, composite cache keys, structural isolation
over filtering).

### Step 1 — `epoch_ingest.py` (Session 15.1, DATA isolation)
- **`FakeFAISSRouter`** — simulates one vector index per `(hospital_id,
  expert_id)`. There is no "search everything" method — every call is
  scoped, so a forgotten filter can't leak data across tenants.
- **`CompositeCheckpointer`** — a LangGraph-checkpointer stand-in keyed on
  `(hospital_id, expert_id, thread_id)`, not `thread_id` alone. A replayed
  or colliding thread ID from another hospital cannot surface someone
  else's state.
- **`FakeRLSDatabase`** — simulates Postgres Row Level Security: every row
  is tagged with `hospital_id` at write time, every read requires one and
  raises `TenantIsolationError` rather than silently returning nothing.
- **FastAPI gateway** — a `/query` route that takes `hospital_id` only from
  a trusted header, exercised via `TestClient` so the module runs and
  self-verifies with no server process required.
- Seeds Dr. Chen's pulmonology corpus under `hospital_a` and a small
  cardiology corpus under `hospital_b` to prove cross-tenant isolation.

### Step 2 — `epoch_auth.py` (Session 15.2, ACTIONS isolation)
- **`mint_jwt` / `verify_jwt`** — a self-contained HMAC-SHA256 JWT
  (issuer/audience/expiry checked, signature verified with
  `hmac.compare_digest`), carrying `hospital_id`, `sub`, and `role`.
  The signing key is `JWT_SECRET`, overridable via `.env` (see
  "Environment variables" above) — it falls back to a hardcoded teaching
  placeholder so the module runs with zero configuration.
- **`ROLE_TOOLS`** — the RBAC matrix (`viewer` / `analyst` / `admin`).
  A role not listed gets zero tools: deny-by-default.
- **`BoundGeminiModel`** — binds a model to *exactly* the tool names
  `ROLE_TOOLS[role]` allows. A disallowed tool call raises `AuthError`
  before it ever reaches `TOOL_REGISTRY` — the schema was never attached,
  not just "the model was told not to."
- **`AUDIT_LOG` / `audit_tool_call`** — every tool invocation is appended
  with actor, hospital, tool name, and timestamp.
- **`epoch_invoke`** — the JWT-verified, role-bound gateway function that
  `epoch_sla.py` wraps in Step 3.

### Step 3 — `epoch_sla.py` (Session 15.3, TIME isolation)
- **`EpochState` / `FakeCheckpointer`** — a LangGraph-style scratchpad
  (`internal_notes`) and checkpoint store, extending 15.1's composite key.
- **`FakeEpochAgent`** — a 7-step simulated agent run. Each step sleeps,
  appends one atomic, standalone note, then checkpoints — so a cancellation
  mid-run always leaves a coherent, replayable partial state behind.
- **`extract_partial_state`** — reads the last checkpoint after a timeout
  and shapes it into a 206 response body (notes, plan-so-far, resume
  token) in well under 50ms — the checkpoint read is cheap, the agent run
  was the expensive part.
- **`CircuitBreaker`** — a 3-state (`CLOSED` / `OPEN` / `HALF_OPEN`)
  sliding-window breaker that fails fast with `503` when the recent
  failure rate crosses threshold, and probes recovery with exactly one
  request after cooldown to avoid a thundering herd.
- **`epoch_invoke_sla`** — wraps `epoch_invoke` with three defense layers:
  circuit breaker (outer) → `asyncio.wait_for(SLA_BUDGET_SECONDS)` (middle)
  → LangGraph `recursion_limit` (inner). JWT verification happens *outside*
  the timeout window so an auth failure never eats SLA budget.
  Returns `200` on completion, `206` with partial notes on a timeout that
  produced ≥1 note, `504` on a timeout that produced zero, `503` when the
  breaker is open.
- **Bug fixed during build:** the original step loop checkpointed a step
  *before* checking the failure-injection condition, so
  `inject_failure_after_steps=0` still checkpointed one note instead of
  zero — silently breaking the "504 on empty timeout" contract. Fixed by
  moving the injection check to the top of the loop, before that step's
  work runs, so `inject_failure_after_steps=N` checkpoints exactly steps
  `0..N-1` as documented.

### Step 4 — `epoch_dashboard.py` (local validation tool, not a graded session)
A single FastAPI process that imports the *real* objects from all three
modules (one shared `FakeCheckpointer`, `CircuitBreaker`, and `AUDIT_LOG`
per process) behind a one-page UI — see Section 3 below.

---

## 3. How to validate the build in the local dashboard

Start it (`python epoch_dashboard.py`, open `http://127.0.0.1:8000`) and
work through the six panels top to bottom — each maps to one guarantee:

| # | Panel | Action | Expected result |
|---|---|---|---|
| 1 | **Mint Identity** | Hospital `hospital_a`, role `analyst` → **Mint JWT** | Token appears; `allowed_tools` shows `search_expert_corpus`, `lookup_patient_history`, `fetch_prior_imaging` (no `prescribe_followup`) |
| 2 | **Data Isolation** | `hospital_a` / `chen` / "GGO protocol" → **Search** | 3 hits from Dr. Chen's corpus |
| 2 | **Data Isolation** | Click **Try wrong hospital →** (`hospital_b` / `chen`) | `hits: []` — proves the FAISS index is scoped per tenant, not filtered |
| 3 | **RBAC Tool Call** | Tool `prescribe_followup` → **Call tool** (still analyst) | `403 tool_not_bound` — the role was never bound to this tool |
| 3 | **RBAC Tool Call** | Re-mint as `admin`, retry `prescribe_followup` | `200`, and a new row appears in the **Audit Log** panel |
| 4 | **SLA Engine** | Scenario "Complete run" → **Run agent** | `200` in ~4.5s (under the 5.0s budget), full `internal_notes` list, `complete: true` |
| 4 | **SLA Engine** | Scenario "Cut cable after 3 steps" → **Run agent** | `206` in ~1.2s, `partial_analysis.internal_notes` has 3 complete sentences, a `resume_token` |
| 4 | **SLA Engine** | Scenario "Cut cable after 0 steps" → **Run agent** | `504` immediately — deny-by-default, no partial data existed to return |
| 5 | **Circuit Breaker** | **Force 10 failures → OPEN** | State badge turns `OPEN`; any `/api/invoke` call now short-circuits to `503` without running the agent |
| 5 | **Circuit Breaker** | Wait 15s (the cooldown), run any SLA scenario | State flips to `HALF_OPEN` for exactly one probe request, then `CLOSED` if it succeeds |
| 6 | **Audit Log** | — | Every tool call from panel 3 (and the tool calls inside SLA runs) appears with actor, hospital, and timestamp |

If every row above matches, all three isolation axes are demonstrably
working — not just passing an assertion in a script, but visibly behaving
correctly end to end.

You can also re-run the non-interactive proof at any time:
```bash
python epoch_ingest.py && python epoch_auth.py && python epoch_sla.py
```
Each prints `N/N checks passed` and a `✓ Session 1X.Y COMPLETE.` line.

---

## 4. Weekly build summary — Week 15

| Session | File | Isolation axis | What it proves | Deny-by-default mechanism |
|---|---|---|---|---|
| **15.1** | `epoch_ingest.py` | **Data** | Hospital A's vectors are structurally unreachable from a Hospital B query | Separate index per `(hospital_id, expert_id)`; RLS raises without a tenant scope |
| **15.2** | `epoch_auth.py` | **Actions** | A role can only call tools it was explicitly bound to, and every call is audited | `ROLE_TOOLS` allowlist; unlisted role/tool → zero tools, not "restricted" tools |
| **15.3** | `epoch_sla.py` | **Time** | A hung or slow agent degrades gracefully (206 + resume token) instead of hanging or erroring away real work | Timeout → partial state if ≥1 note exists, else 504; breaker fails fast under sustained failure |
| — | `epoch_dashboard.py` | (all three) | Click-through validation surface reusing the real objects from all three modules | n/a — it's a viewer, not a new isolation layer |

**The throughline:** every layer treats isolation as a *structural*
property (a separate index, a key that was never bound, a budget that was
never granted) rather than a *filter* applied after the fact (a `WHERE`
clause that could be forgotten, a prompt instruction that could be
ignored, a timeout that could silently return an empty error). That's the
"triangle" the code refers to as closed at the end of 15.3: DATA · ACTIONS
· TIME.

---

## 5. What's coming next

The end of `epoch_sla.py`'s own output names the next step directly:

> **Session 16.1 — Containerise EPOCH as a 6-service Docker cluster.**
> The isolation, RBAC, and SLA layers ship as one `docker compose up`.

Concretely, that means the `Fake*` in-memory simulators built across 15.1–
15.3 (`FakeFAISSRouter`, `FakeRLSDatabase`, `FakeCheckpointer`) get swapped
for the real services they stand in for — a real vector index, a real
Postgres instance with RLS policies enabled, a real LangGraph
`AsyncSqliteSaver` (or equivalent) — each running as its own container,
wired together so the same isolation guarantees demonstrated locally here
hold across process and network boundaries, not just within one Python
process. Everything built this week (the composite cache keys, the
`ROLE_TOOLS` matrix, the SLA budget and circuit breaker) carries forward
unchanged — 16.1 is a deployment target for this code, not a rewrite of it.
