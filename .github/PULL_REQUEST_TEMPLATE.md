## What & why

Briefly: what this changes and the motivation.

## Checklist

- [ ] `pytest -q` green
- [ ] `ruff check src tests` + `ruff format --check src tests` clean
- [ ] New behavior has unit tests; coverage not regressed
- [ ] Core stays numpy-only (heavy deps behind an extra, lazy-imported)
- [ ] Updated `spec/*.md` + `CHANGELOG.md` if shipped behavior changed
- [ ] Conventional Commit message (`feat:` / `fix:` / `docs:` / …)

## Notes

Anything reviewers should know — tradeoffs, follow-ups, out-of-scope items.
