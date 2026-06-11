# Contributing to Chronotope

Thanks for your interest. Chronotope is pre-1.0 and moving fast — the API may
change between minor releases. Issues, bug reports, and focused PRs are welcome.

## Dev setup

```bash
git clone https://github.com/wikieden/tempomem.git
cd tempomem
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # core is numpy-only; dev adds pytest/ruff
```

Optional backends (heavy, opt-in):

```bash
pip install -e ".[vec]"          # sqlite-vec ANN index
pip install -e ".[clip]"         # OpenClipEncoder (torch + open-clip)
pip install -e ".[all]"          # everything
```

## Run the checks

```bash
pytest -q                        # unit tests (vec/clip suites self-skip if the extra is absent)
ruff check src tests             # lint
ruff format src tests            # format
python examples/01_quickstart.py # smoke
```

CI runs lint + the unit matrix (3.10–3.12 × macOS/Linux) plus `[clip]` and
`[vec]` lanes. Keep the suite green and coverage from regressing.

## Conventions

- **Python ≥ 3.10**, type hints on all public signatures, PEP 8 via ruff.
- **Core stays numpy-only.** Anything pulling torch/CUDA/native wheels goes
  behind an extra (`[clip]`, `[vec]`, `[perception]`) and lazy-imports, so
  `import tempomem` never drags heavy deps.
- **Tests first for new behavior.** Each feature lands with unit tests; the
  fusion path must stay deterministic (`tests/unit/test_fusion.py`).
- **The `.smem` BLOB is the source of truth.** Indexes like `node_vec` are
  rebuildable mirrors — maintain them on write, never read-authoritative.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`). One logical change per commit.

## Where things live

| Path | What |
|---|---|
| `src/tempomem/` | the library |
| `spec/` | normative API / schema / algorithm specs — update when behavior changes |
| `docs/` | product + roadmap + open questions |
| `docs/sprint/` | per-milestone task breakdowns |
| `examples/` | runnable, no-GPU demos |

When you change shipped behavior, update the matching `spec/*.md` and
`CHANGELOG.md` in the same PR. Design decisions get logged in `docs/05-OPEN.md`.

## License

By contributing you agree your contributions are licensed under Apache-2.0, the
project license.
