# Sprint 00 · Skeleton (M0)

**Goal:** A clean-install `import spatialmem` works on Mac (no CUDA), schema + dataclasses round-trip, and a 50-line fake-detections demo runs green. No real perception. No fusion intelligence yet — just the rails.

**Exit criteria (from [roadmap](../03-ROADMAP.md)):** `pytest -q` green; `import spatialmem` on a clean Python 3.11 venv works; demo notebook produces a queryable store.

## Task Breakdown

| ID | Task | Output | Depends on | Est (CC) |
|---|---|---|---|---|
| T1 | `pyproject.toml` — hatch, deps (numpy/scipy/sqlite-vec/pillow/pydantic), extras stubs, ruff+pyright config | installable skeleton | — | 20 min |
| T2 | Package skeleton `src/spatialmem/__init__.py` with `__all__` + version | `import spatialmem` works | T1 | 10 min |
| T3 | `frame.py` — `Detection`, `Observation` frozen dataclasses + JSON round-trip | typed value objects | T2 | 30 min |
| T4 | `persist/schema.sql` + `persist/migrations/001_init.py` — all tables from [SCHEMA.md](../../spec/SCHEMA.md) | empty store creatable | T2 | 40 min |
| T5 | `persist/__init__.py` — open/create, sqlite-vec + rtree load, migration runner, WAL | `SpatialMemory.open()` returns a live store | T4 | 40 min |
| T6 | `store.py` — Node/Edge/Episode CRUD + `stats()` | read/write graph rows | T5, T3 | 50 min |
| T7 | Fusion **stub** — every observation = new node (no merge logic yet) | observations land as nodes | T6 | 20 min |
| T8 | `query.py` minimal — `recent()` + `spatial()` (R-tree), no semantic yet | nodes retrievable | T6 | 30 min |
| T9 | `serialize.py` — `format="json"` + basic `format="prompt"` | graph → text | T6 | 30 min |
| T10 | `cli.py` — `spatialmem inspect <file>` | counts + sample nodes | T6 | 20 min |
| T11 | `examples/01_quickstart.py` — synthetic kitchen detections in, query out | runnable demo | T7, T8, T9 | 20 min |
| T12 | Tests: schema round-trip, store CRUD, JSON round-trip, demo-as-test | `pytest -q` green | T3–T9 | 40 min |
| T13 | CI: `.github/workflows/ci.yml` — lint + unit on 3.10/3.11/3.12 × mac/linux | green badge | T12 | 30 min |

Total ~6 h CC time.

## Sequencing

```
T1 → T2 → T3 ─┐
              ├→ T4 → T5 → T6 → T7 ─┐
              │                T8 ──┤
              │                T9 ──┤→ T11 → T12 → T13
              │               T10 ──┘
```

T3 and T4 can run in parallel after T2. T7/T8/T9/T10 all fan out from T6.

## Not in this sprint

- Real fusion scoring (geom/iou/sem/label) → M1
- Semantic retrieval / CLIP → M1
- ConceptGraphs adapter → M2
- LLM verbalizer → M2

## Definition of Done

- [ ] `pip install -e .` on clean Python 3.11 venv, no CUDA, < 25 s
- [ ] `python examples/01_quickstart.py` prints a query hit
- [ ] `spatialmem inspect kitchen.smem` shows node counts
- [ ] `pytest -q` green; coverage reported (gate enforced from M1)
- [ ] CI green on both OS
- [ ] B2 + B4 resolved (name check, license) — see [05-OPEN.md](../05-OPEN.md)

## Risks

| Risk | Mitigation |
|---|---|
| sqlite-vec wheel missing for an OS/Python combo | Verify wheel matrix in T1 before committing to the dep |
| R-tree (`pysqlite` rtree module) not compiled in stdlib sqlite | Detect at `open()`, raise clear error; document min sqlite version |
| Scope creep into real fusion during T7 | T7 is explicitly a stub — merge logic is M1, enforce in review |
