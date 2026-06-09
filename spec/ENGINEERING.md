# spec · Engineering Standards

Normative for all contributors, including future-me.

## Repo Layout

```
spatialmem/                ← repo root
├── README.md
├── LICENSE                  Apache-2.0
├── pyproject.toml           PEP 621 + hatch backend
├── docs/                    product + design decisions
├── spec/                    normative specs (this dir)
├── src/
│   └── spatialmem/
│       ├── __init__.py
│       ├── frame.py
│       ├── store.py
│       ├── fusion.py
│       ├── persist/
│       │   ├── __init__.py
│       │   ├── schema.sql
│       │   └── migrations/
│       │       └── 001_init.py
│       ├── query/
│       │   ├── __init__.py
│       │   ├── router.py
│       │   ├── retrievers.py
│       │   └── verbalize.py
│       ├── serialize.py
│       ├── adapters/
│       │   ├── detections.py
│       │   └── conceptgraphs.py     # extras=[conceptgraphs]
│       ├── bridges/
│       │   ├── ros2.py              # extras=[ros2]
│       │   └── mem0.py              # extras=[mem0]
│       ├── llm/
│       │   ├── protocol.py
│       │   ├── openai.py
│       │   └── ollama.py
│       └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── golden/              recorded inputs + expected SQLite SHA-256
├── examples/
│   ├── 01_quickstart.py
│   ├── 02_replica_scan.py
│   └── 03_ros2_bridge/
└── benchmarks/
    └── bench_ingest_query.py
```

## Language & Tooling

- Python **≥ 3.10**; CI matrix 3.10 / 3.11 / 3.12.
- Build: **hatch**.
- Lint + format: **ruff** (`select = ["E","F","I","B","UP","SIM","TID","PERF","RUF"]`), line length 100.
- Type-check: **pyright** strict on `src/spatialmem/`. `# type: ignore` requires a `# reason: ...` suffix.
- Tests: **pytest**, **pytest-cov**, **hypothesis** for property tests on fusion.
- Pre-commit: ruff, pyright, end-of-file fixer, no large files.
- Docs: built docs (M3+) via **mkdocs-material**.

## Package Hygiene

- Public API surface lives in `src/spatialmem/__init__.py` `__all__`. Anything not listed is private.
- Default install must not pull Torch, CUDA, ROS, or any binary > 50 MB.
- Optional features ship as PEP 508 extras: `[clip]`, `[conceptgraphs]`, `[ros2]`, `[mem0]`, `[viz]`, `[all]`.
- Native extensions only via well-known wheels (sqlite-vec, etc.). No build-from-source dependencies in default install.

## Coding Conventions

- Dataclasses (`frozen=True, slots=True`) for value types. Pydantic only at user-facing config boundary.
- No global mutable state. `SpatialMemory` is the only stateful object.
- Logging via `logging.getLogger("spatialmem.<module>")`. Never `print()`.
- Public methods document units + frames in their docstring. Repeat even if redundant — these are the bugs we will hit.
- Returns over raises for expected absence (`Optional`, `[]`). Raise only for programmer error or IO failure.
- No comments narrating *what* the code does. Comments only for *why*, with a referenced spec section.

## Auto-flush (`max_pending_obs`)

`add_detections()` only *stages* observations into `_pending`; `commit()` is the
sole drainer that fuses them (fuse-before-persist). A long-running ingest loop
that forgets to `commit()` would let `_pending` grow without bound.
`SpatialMemConfig.max_pending_obs` is an optional guard: set to `N >= 1`,
`add_detections()` calls `commit()` automatically once `len(_pending) >= N`,
emitting a `WARNING` on the `spatialmem.memory` logger. The auto-flush routes
through the *same* `commit()`, so fuse-before-persist still holds (no
orphan/unfused rows) and every observation keeps its per-call `episode` binding.
`None` (default) disables the guard — the caller owns `commit()`. A threshold
`< 1` is rejected at config construction.

## Testing

| Layer | Required |
|---|---|
| Unit | Every module in `fusion`, `store`, `query`, `serialize` has ≥ 1 unit test per public function. |
| Property | `hypothesis` strategies for `Detection` → arbiter never crashes, conserves observation count. |
| Golden | Recorded detection stream + frozen SQLite SHA-256 + frozen query top-k. Determinism guard. |
| Integration | ConceptGraphs adapter against a 10-frame snippet (CI artifact). |
| Bench | `pytest-benchmark` gate: ingest TPS, query p95. |
| Doc | `pytest --doctest-modules`; README quickstart runs as a test. |

Coverage gate: ≥ 75% on core modules, measured by `pytest-cov` in CI. PRs lowering coverage are blocked.

## CI (GitHub Actions)

Jobs (run in parallel where possible):

1. `lint` — ruff + pyright strict.
2. `unit` — `pytest -q tests/unit/` on 3.10 / 3.11 / 3.12 × macOS arm64 / Ubuntu x86_64.
3. `integration` — `pytest tests/integration/` (Ubuntu only, ConceptGraphs extras).
4. `bench-gate` — runs nightly; opens issue on regression > 20%.
5. `package` — `hatch build`, sanity-import in clean venv on both OS.
6. `docs-readme` — runs README quickstart as a script.

A PR cannot merge with any required check red.

## Versioning & Release

- SemVer. `0.x.y` until M3 ships and we lock the stable API surface.
- `CHANGELOG.md` in keep-a-changelog format, updated in the same PR as the change.
- Tags `v0.1.0`, `v0.1.1`, … trigger PyPI publish via Trusted Publisher (no API token).
- Release notes: human-written, copied from CHANGELOG.

## Branch & Commit

- Trunk-based. Branch off `main`, PR back. No long-lived feature branches.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `bench:`, `chore:`).
- Squash-merge by default; preserve PR title as commit subject.
- One logical change per PR. If you need a setup commit + behavior commit, that's two PRs.

## Security

- No telemetry. No network calls in core. Adapters that hit external APIs (LLM, hosted services) require explicit user config.
- Dependabot weekly. CodeQL on push to main.
- Secrets only in `.env.local` (gitignored). Examples ship a `.env.example`.
- SBOM emitted at release (`pip-audit --format=cyclonedx-json`).

## Performance Budgets (CI-enforced)

| Operation | Hardware | Budget |
|---|---|---|
| `add_detections([d])` | M2 Air | < 3 ms p50 |
| `commit()` over 100 obs | M2 Air | < 30 ms p50 |
| `query(text)` over 10k nodes | M2 Air | < 150 ms p95 |
| Cold `open()` on 100 MB store | M2 Air | < 200 ms |
| `pip install spatialmem` | clean venv | < 25 s |

If a PR regresses any budget > 20%, CI blocks. Hot-fixing a regression by raising a budget requires sign-off in the PR description.

## What we will not do

- Vendor a C/C++ extension we don't own.
- Add a hard dependency on any single LLM or vector DB vendor.
- Ship telemetry "for product improvement."
- Use `print`, `pickle`, or pre-3.10 typing syntax.
