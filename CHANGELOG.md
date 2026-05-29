# Changelog

All notable changes to SpatialMem are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer
(pre-1.0, the API may change between minor releases).

## [Unreleased] — M2 in progress

### Added
- **`decay(half_life_days, min_conf)`** — age-decays node confidence
  (`conf' = conf · 0.5^(age/half_life)` from `t_last`) and prunes nodes below
  the floor. Memory hygiene for long-lived stores.
- **`answer(query, verbalizer=...)`** — retrieve → serialize scene prompt →
  BYO LLM. `Verbalizer` protocol (wrap OpenAI / Anthropic / Ollama); no bundled
  model or key. `SpatialMemory.open(verbalizer=...)` or per-call.
- **`spatialmem.bench.recall_at_k`** — lightweight retrieval eval over scripted
  `(query, expected_label)` cases; powers the M2 demo metric.
- **`[clip]` CI lane** — installs the extra and smoke-tests `OpenClipEncoder`
  shape/dim against real Torch (random init, no weight download).
- **sqlite-vec ANN** (`[vec]` extra) — a `node_vec` vec0 cosine index mirrors
  node features, maintained on insert/update/delete and queried by
  `semantic_vec`. Falls back to the linear scan when the extension is absent;
  the BLOB feature stays the source of truth. `[vec]` CI lane added.
- **`resplit()`** (split detection) — a node whose member observations form two
  clusters separated by more than `tau_split_m` (each ≥ `min_split_obs`) is
  split back into two nodes (deterministic 2-means over observation centroids).
- **`PerceptionAdapter` protocol + `add_frame(rgb, depth, pose)`** — the RGB-D
  seam. `add_frame` routes a posed frame through a configured adapter to
  detections, then fuses. Any backend or test stub plugs in; the concrete
  ConceptGraphs adapter lands behind a `[perception]` extra (needs CUDA).

### Notes
- Perception backend decided: `PerceptionAdapter` protocol + ConceptGraphs
  (SAM/GroundingDINO/OpenCLIP) as first adapter; no NVIDIA model as default
  (license traps). See `docs/05-OPEN.md` P1–P3.
- V-track complete (V1–V6) + split detection (V4) + perception seam (P0/P2).
  Remaining M2: concrete ConceptGraphs adapter (P1) + bench parity (P3) — both
  need a CUDA dev box; the protocol seam and `add_frame` wiring are done and
  tested via a stub adapter.

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
