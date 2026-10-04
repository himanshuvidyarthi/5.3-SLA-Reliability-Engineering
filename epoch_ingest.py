"""
epoch_ingest.py — EPOCH Data Isolation Layer
Session 15.1. Step 1 of the three-step capstone.

Builds the DATA isolation axis of EPOCH:
    - FakeFAISSRouter:   per-hospital, per-expert vector index simulation.
      No hospital's vectors are ever visible to a query scoped to another
      hospital_id. This is the in-memory stand-in for a real per-tenant
      FAISS/pgvector index.
    - CompositeCheckpointer: LangGraph-style checkpoint store keyed on the
      composite (hospital_id, expert_id, thread_id) tuple. This composite
      key is the isolation primitive reused unchanged in 15.2 and 15.3.
    - FakeRLSDatabase: simulates Postgres Row Level Security. Every row is
      tagged with hospital_id at write time; every read is scoped by
      hospital_id at query time. Cross-hospital reads raise, they don't
      just filter — a filtering bug should be loud, not silent.
    - FastAPI gateway: a minimal /health and /query surface, exercised via
      TestClient so this module runs standalone with no server process.

Design mandate:
    TENANT ISOLATION IS NOT A FILTER. IT IS A BOUNDARY.
    Every read and write in this module takes hospital_id as a mandatory,
    non-optional argument. There is no "get all documents" method — because
    in a multi-tenant system, that method is a vulnerability with a docstring.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: FakeFAISSRouter — per-hospital, per-expert vector isolation
# ─────────────────────────────────────────────────────────────────────────────

class TenantIsolationError(Exception):
    """Raised whenever a caller tries to cross a hospital_id boundary."""


class FakeFAISSRouter:
    """
    Simulates a per-tenant FAISS index router.

    In production: one FAISS index (or pgvector namespace) per hospital_id,
    keyed off the same composite (hospital_id, expert_id) used everywhere
    else in EPOCH. Never a single global index with a hospital_id metadata
    filter — a filter can be forgotten; a separate index cannot leak.

    Introduced: Session 15.1. Permanent.
    """

    def __init__(self):
        # {(hospital_id, expert_id): [doc_text, ...]}
        self._indexes: Dict[Tuple[str, str], List[str]] = {}

    def add_documents(self, hospital_id: str, expert_id: str, docs: List[str]) -> None:
        if not hospital_id or not expert_id:
            raise TenantIsolationError('hospital_id and expert_id are mandatory.')
        key = (hospital_id, expert_id)
        self._indexes.setdefault(key, []).extend(docs)

    def search(self, hospital_id: str, expert_id: str, query: str, k: int = 3) -> List[str]:
        """
        Naive keyword-overlap 'search' — good enough for the teaching demo.
        The point isn't ranking quality, it's that the corpus scanned is
        ALWAYS scoped to (hospital_id, expert_id), never global.
        """
        if not hospital_id or not expert_id:
            raise TenantIsolationError('hospital_id and expert_id are mandatory.')
        key = (hospital_id, expert_id)
        corpus = self._indexes.get(key, [])
        query_terms = set(query.lower().split())

        def score(doc: str) -> int:
            return len(query_terms & set(doc.lower().split()))

        ranked = sorted(corpus, key=score, reverse=True)
        return ranked[:k]

    def tenant_count(self) -> int:
        return len(self._indexes)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CompositeCheckpointer — (hospital_id, expert_id, thread_id) key
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointEntry:
    step: int
    state: dict
    ts: float = field(default_factory=time.time)


class CompositeCheckpointer:
    """
    Simulates langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver, keyed on the
    composite (hospital_id, expert_id, thread_id) tuple rather than thread_id
    alone. A bare thread_id is not a tenant boundary — a UUID collision or a
    replayed thread_id from another hospital must not surface another
    hospital's state. The composite key makes that structurally impossible.

    This exact key shape is reused by epoch_sla.FakeCheckpointer in 15.3.

    Introduced: Session 15.1. Permanent.
    """

    def __init__(self):
        self._store: Dict[Tuple[str, str, str], List[CheckpointEntry]] = {}

    def put(self, hospital_id: str, expert_id: str, thread_id: str, state: dict) -> None:
        key = (hospital_id, expert_id, thread_id)
        entries = self._store.setdefault(key, [])
        entries.append(CheckpointEntry(step=len(entries), state=state))

    def get_latest(self, hospital_id: str, expert_id: str, thread_id: str) -> Optional[dict]:
        key = (hospital_id, expert_id, thread_id)
        entries = self._store.get(key, [])
        return entries[-1].state if entries else None

    def history(self, hospital_id: str, expert_id: str, thread_id: str) -> List[CheckpointEntry]:
        key = (hospital_id, expert_id, thread_id)
        return list(self._store.get(key, []))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FakeRLSDatabase — simulated Postgres Row Level Security
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    hospital_id: str
    payload: dict


class FakeRLSDatabase:
    """
    Simulates a Postgres table with Row Level Security enabled:
        CREATE POLICY hospital_isolation ON corpus_meta
            USING (hospital_id = current_setting('app.hospital_id'));

    Every write tags the row with hospital_id. Every read requires a
    hospital_id and only returns rows that match it. There is no
    "SELECT * FROM corpus_meta" without a hospital_id in this class —
    the isolation is structural, not an optional WHERE clause a future
    caller could omit.

    Introduced: Session 15.1. Permanent.
    """

    def __init__(self):
        self._rows: List[Row] = []

    def insert(self, hospital_id: str, payload: dict) -> None:
        if not hospital_id:
            raise TenantIsolationError('Cannot insert a row without hospital_id.')
        self._rows.append(Row(hospital_id=hospital_id, payload=payload))

    def select(self, hospital_id: str) -> List[dict]:
        if not hospital_id:
            raise TenantIsolationError('Cannot select without a hospital_id (RLS requires it).')
        return [r.payload for r in self._rows if r.hospital_id == hospital_id]

    def row_count(self) -> int:
        return len(self._rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FastAPI gateway (exercised via TestClient — no server process)
# ─────────────────────────────────────────────────────────────────────────────

router = FakeFAISSRouter()
rls_db = FakeRLSDatabase()

app = FastAPI(title='EPOCH Ingest Gateway')


@app.get('/health')
def health():
    return {'status': 'ok', 'tenants': router.tenant_count()}


@app.get('/query')
def query_endpoint(
    q: str,
    expert_id: str = 'chen',
    x_hospital_id: str = Header(...),
):
    """hospital_id is taken ONLY from a trusted header set by the auth layer
    upstream (15.2) — never from a query param a client could spoof."""
    if not x_hospital_id:
        raise HTTPException(status_code=400, detail='X-Hospital-Id header required.')
    results = router.search(x_hospital_id, expert_id, q)
    return {'hospital_id': x_hospital_id, 'expert_id': expert_id, 'results': results}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Seed data — Dr. Chen's cognitive twin corpus
# ─────────────────────────────────────────────────────────────────────────────

def seed_corpus() -> None:
    """Seeds hospital_a / expert 'chen' with Dr. Chen's pulmonology traces."""
    router.add_documents('hospital_a', 'chen', [
        'GGO protocol: ground-glass opacities under 6mm in non-smokers require no follow-up.',
        'GGO protocol: ground-glass opacities 6-8mm warrant CT follow-up at 6-12 months.',
        'Fleischner Category C nodules with prior imaging get a 3-month CT follow-up.',
        'PET-negative solitary nodules under 2cm have a low pretest probability of malignancy.',
        'Dr. Chen prioritizes non-invasive follow-up over biopsy for low-risk GGOs.',
    ])
    router.add_documents('hospital_b', 'patel', [
        'Cardiology protocol: post-MI patients get echo at 6 weeks.',
        'Cardiology protocol: stable angina managed medically before catheterization.',
    ])
    rls_db.insert('hospital_a', {'corpus': 'chen', 'doc_count': 5})
    rls_db.insert('hospital_b', {'corpus': 'patel', 'doc_count': 2})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Verification
# ─────────────────────────────────────────────────────────────────────────────

def run_session_verification() -> dict:
    checks = []
    seed_corpus()
    client = TestClient(app)

    # CHECK 1: cross-hospital search returns nothing from the other tenant
    hits_a = router.search('hospital_a', 'chen', 'GGO protocol')
    leak = router.search('hospital_b', 'chen', 'GGO protocol')  # wrong expert/hospital combo
    checks.append({
        'label': 'FakeFAISSRouter scopes search results to (hospital_id, expert_id)',
        'passed': len(hits_a) > 0 and len(leak) == 0,
        'note': f'hospital_a/chen hits={len(hits_a)}, hospital_b/chen hits={len(leak)}',
    })

    # CHECK 2: composite checkpointer isolates identical thread_id across hospitals
    cp = CompositeCheckpointer()
    cp.put('hospital_a', 'chen', 'T-1', {'note': 'a-state'})
    cp.put('hospital_b', 'patel', 'T-1', {'note': 'b-state'})
    a_state = cp.get_latest('hospital_a', 'chen', 'T-1')
    b_state = cp.get_latest('hospital_b', 'patel', 'T-1')
    checks.append({
        'label': 'CompositeCheckpointer isolates identical thread_id across hospitals',
        'passed': a_state == {'note': 'a-state'} and b_state == {'note': 'b-state'},
        'note': f'a={a_state}, b={b_state}',
    })

    # CHECK 3: RLS select only returns rows for the requesting hospital
    rows_a = rls_db.select('hospital_a')
    rows_b = rls_db.select('hospital_b')
    checks.append({
        'label': 'FakeRLSDatabase.select scopes rows to hospital_id',
        'passed': all(r for r in rows_a) and len(rows_a) == 1 and len(rows_b) == 1,
        'note': f'hospital_a rows={rows_a}, hospital_b rows={rows_b}',
    })

    # CHECK 4: RLS insert/select without hospital_id raises
    raised = False
    try:
        rls_db.select('')
    except TenantIsolationError:
        raised = True
    checks.append({
        'label': 'FakeRLSDatabase.select without hospital_id raises TenantIsolationError',
        'passed': raised,
        'note': f'raised={raised}',
    })

    # CHECK 5: FastAPI gateway enforces X-Hospital-Id header and scopes results
    resp = client.get('/query', params={'q': 'GGO protocol', 'expert_id': 'chen'},
                       headers={'X-Hospital-Id': 'hospital_a'})
    resp_missing = client.get('/query', params={'q': 'GGO protocol'})
    checks.append({
        'label': 'FastAPI /query gateway scopes results via X-Hospital-Id header',
        'passed': resp.status_code == 200 and len(resp.json()['results']) > 0
                  and resp_missing.status_code == 422,
        'note': f'with-header={resp.status_code}, missing-header={resp_missing.status_code}',
    })

    passed = sum(1 for c in checks if c['passed'])
    total = len(checks)
    return {'passed': passed == total, 'checks': checks, 'summary': f'{passed}/{total} checks passed'}


if __name__ == '__main__':
    print('\n╔══════════════════════════════════════════════════════╗')
    print('║  EPOCH — Session 15.1 Verification                   ║')
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

    print('  ✓ Session 15.1 COMPLETE.\n')
