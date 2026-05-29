# 05 · Open Questions

Live design decisions not yet locked. Each has an owner-decision deadline (the milestone by which it must be resolved or it blocks). Resolved entries move to a dated decision log at the bottom.

## Fusion

| # | Question | Why it matters | Decide by |
|---|---|---|---|
| F1 | Sync per-frame vs batched-at-commit fusion? | Latency vs throughput; affects API contract of `add_*` return value | M1 |
| F2 | Static vs dynamic object handling — a mug moves, a fridge doesn't | Wrong = either ghost duplicates (treat moved mug as new) or stale positions | M2 |
| F3 | Per-class threshold overrides (`τ_merge` per label)? | Global threshold mis-merges small objects; per-class needs a config surface | M2 |
| F4 | How are feature centroids kept stable as a node accrues 100s of obs? | EMA drift can pull a node's embedding off its true identity | M2 |
| F5 | Split detection default — on or off at M2? | Off = blobs merge; on = compute cost + false splits | M2 |

## Schema / Storage

| # | Question | Why it matters | Decide by |
|---|---|---|---|
| S1 | One `.smem` file vs file-per-episode? | Multi-session robots may want episode isolation + cheap pruning | M3 |
| S2 | Embedding dim locked at `open` — migration path if user switches CLIP→SigLIP? | Re-embedding a whole store is expensive; need a re-index tool | M2 |
| S3 | Store raw point clouds or only aggregated geometry? | Raw = split/re-fusion possible later but 10–100× disk | M1 |
| S4 | sqlite-vec vs faiss-on-disk above ~1M vectors? | sqlite-vec recall/latency unverified at scale | M3 |

## Query

| # | Question | Why it matters | Decide by |
|---|---|---|---|
| Q1 | Intent heuristic locales beyond en/zh? | Non-en/zh users get semantic-only fallback (degraded) | M3 |
| Q2 | LLM verbalizer — bundled default model or strictly BYO? | Bundled = better demo, adds a network/secret dependency | M2 |
| Q3 | How to score recency vs confidence in ranking when they conflict? | "Last seen" vs "most confident" can disagree; user expectation unclear | M1 |

## Perception backend

| # | Question | Why it matters | Decide by |
|---|---|---|---|
| P1 | ConceptGraphs pinned commit — track upstream or hard-fork? | Upstream is research-paced; drift breaks our adapter | M2 |
| P2 | Hydra adapter — worth it given ROS dependency, or skip to nvblox-features? | Hydra gives hierarchy free but drags ROS into the extra | M3 |
| P3 | Do we ship a default detector (SAM+CLIP) in `[clip]`, or detections-only forever in core? | Detections-only keeps core tiny but raises the "hello world" bar | M2 |

## Product / brand

| # | Question | Why it matters | Decide by |
|---|---|---|---|
| B1 | Flip public at M0 (docs only) or M1 (runnable)? | Currently private. Empty-ish repo public = weak first impression | M1 |
| ~~B2~~ | ~~Name conflict check~~ — **resolved 2026-05-29, see decision log** | | done |
| B3 | Mem0 — reach out for partnership before or after launch? | Pre-launch goodwill vs showing traction first | M3 |
| B4 | License Apache-2.0 confirmed vs MIT (eMEM is MIT)? | Patent grant (Apache) vs maximal permissiveness (MIT) | M0 |

## Decision Log (resolved)

| Date | # | Decision | Rationale |
|---|---|---|---|
| 2026-05-29 | — | MVP is detections-in only, perception deferred to M2 | Win the API first; perception install pain must not gate "hello world" |
| 2026-05-29 | — | Single `.smem` SQLite file as the unit | Portable, diffable, no external services |
| 2026-05-29 | — | Repo private until runnable (M0/M1) | "private first" — avoid weak empty-repo first impression |
| 2026-05-29 | B2 | Name "spatialmem" confirmed clear | PyPI free (`spatialmem`/`spatial-mem`/`spatialmemory` all 404); GitHub matches are unrelated (Julia `.jl` 1★, rest 0★ dormant) — no Python pkg or brand collision |
