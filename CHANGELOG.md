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
- **Read-only HTML viewer** — `spatialmem viz store.smem -o scene.html` (and
  `spatialmem.viz.to_html`). Self-contained: top-down 2D scatter of node
  centroids + a node table, no JS framework / network. (Deferred M1 web viewer.)
- **Dataset streaming** — `DatasetSource` protocol + `stream(mem, source)` +
  `SyntheticScene` (deterministic multi-frame GT scene) + `HashEncoder` fixture.
  Demonstrates incremental fusion: a scene streamed frame-by-frame converges
  many observations to one node per object (no GPU). See `examples/03`.
- **Hierarchy / rooms** — `define_region(label, bbox_min, bbox_max, *,
  type_="room")` creates a region node and adopts every object whose centroid
  falls inside it as a child (`parent_id`). Region feature = `encoder.encode_text(label)`
  when an encoder is set (so `query("kitchen")` ranks it), else the mean of child
  features. `contents(region)` returns a region's children (by id or label).
  `serialize(format="prompt")` now nests objects under their region; the fusion
  arbiter skips non-object nodes so observations never merge into a region.
  Idempotent by `(label, type_)`; `forget`/`decay` reparent children safely and
  leave regions intact (regions are structure, not observations).
- **Spatial relations** — `relate()` infers `near` / `on` / `under` edges
  between object nodes from geometry (centroid distance + bbox stacking), no
  perception. `related(node, rel=None)` returns a node's neighbors as
  `(NodeHit, relation_type)`, by node id or object label. Idempotent recompute.
- **`update()` + `history()`** — `update(node_id, *, label=, center_xyz=,
  confidence=)` corrects a node in place (Mem0-style; moving the centroid shifts
  the bbox, keeping extent). `history(node_id)` returns the time-ordered
  observation trail behind a node — every sighting with its timestamp and
  position, so "where was it over time" / "last seen" is `history(id)[-1]`.
- **Relation-aware prompt** — `serialize(format="prompt")` now appends each
  node's relation edges as a `| on table#3, near kettle#2` suffix (after
  `relate()`), so an LLM via `answer()` sees the scene graph, not just an object
  list. Toggle with `serialize(..., relations=False)`.
- **Multi-session merge** — `merge(other_smem)` folds another store's objects
  into this one through fusion: the same physical object seen in a later session
  converges to one node, new objects are added. Re-entering a space continues
  the memory instead of starting over. Dim-checked; regions are not merged.
- **Relational NL query** — `query("what's on the table")` now parses a relation
  phrase (`on` / `near` / `under` / `next to` / `beside` …) + an anchor object
  and traverses the relation edges, so the relations from `relate()` are
  answerable in natural language. Falls back to semantic/keyword when no
  relation phrase or anchor object matches.
- **Change detection** — `moved(node_id)` returns how far an object travelled
  across its observation trail (first→last). `changes(since_ts)` returns a
  `ChangeSet(new, seen_again)` of nodes that first appeared or were re-observed
  since a time. `stale(before_ts)` lists nodes not seen since then (candidate
  "gone"). Temporal memory: "what moved / is new / hasn't been seen".
- **Token-budgeted prompt** — `serialize(format="prompt", max_tokens=N)` caps
  the output: nodes are emitted most-recent-first and the remainder dropped with
  an explicit `… (N more omitted)` marker (never silent). Keeps the LLM payload
  bounded on large scenes.

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
