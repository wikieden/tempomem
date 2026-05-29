# Changelog

All notable changes to SpatialMem are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer
(pre-1.0, the API may change between minor releases).

## [0.1.0a1] - 2026-05-29

### Added
- **Real fusion arbiter** (M1) — replaces the M0 node-per-observation stub.
  Each observation is scored against nearby nodes (geometry + 3D IoU + feature
  cosine + label compatibility) and merged into the best match above
  `tau_merge`, created as a new node, or rejected below `tau_obs`. Two
  sightings of the same object now converge to one node with aggregated
  geometry/feature/label distribution. See `spec/FUSION-ARBITER.md`.
- `FusionConfig` / `SpatialMemConfig` — tunable thresholds and score weights,
  passed via `SpatialMemory.open(config=...)`.
- `store.candidates_near`, `store.update_node` — proximity query + merge update.
- `fusion.iou3d`, `fusion.label_compat`, `fusion.score` — scoring primitives.
- **Semantic query** via a BYO `Encoder` protocol. `SpatialMemory.open(encoder=...)`
  enables `query("red mug")` / `mem.semantic(text)` to embed the query string and
  rank nodes by feature cosine (linear scan); falls back to label keyword match
  when no encoder is set. `OpenClipEncoder` reference impl ships behind the
  `[clip]` extra (lazy Torch import — core stays numpy-only).

### Notes
- Fusion is deterministic for a fixed config + observation stream (tie-break on
  node id). Covered by `tests/unit/test_fusion.py::test_determinism`.
- Semantic ANN (sqlite-vec) and a CLIP text/image encoder remain deferred to a
  later milestone; label compatibility is lexical for now.

## [0.0.1] - 2026-05-29

### Added
- M0 skeleton: installable package (`numpy`-only default dep), `Detection` /
  `Observation` value objects, single-file `.smem` SQLite store with
  forward-only migrations, Node/Edge/Episode CRUD, temporal + spatial +
  keyword retrieval, JSON + prompt serialization, `spatialmem inspect` CLI.
- Synthetic-kitchen quickstart, 18 tests, CI matrix (3.10/3.11/3.12 × mac/linux).
