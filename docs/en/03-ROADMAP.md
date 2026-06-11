> 🌐 **English** · [中文](../zh/03-ROADMAP.md)

# 03 · Roadmap

Milestones are time-boxed, not feature-locked. If a milestone slips on scope, cut scope, don't slip time.

**Status (2026-06-08):** M0 ✅ · M1 ✅ (`v0.1.0a1`) · M2 🟡 in progress (V-track + perception seam done; concrete RGB-D adapter + recorded demo blocked on a CUDA box). 155 tests across the workspace (core 120 / brain 19 / perception 16), core install numpy-only. Per-milestone sprint breakdowns: [SPRINT-00](sprint/SPRINT-00.md) · [SPRINT-01](sprint/SPRINT-01.md) · [SPRINT-02](sprint/SPRINT-02.md).

Legend: ✅ done · 🟡 partial · ⛔ blocked · ⬜ not started.

## M↔P mapping (milestone view ↔ system vision)

This file uses single-package **M-numbering** (M0–M4). The system vision
[docs/VISION.md](../../../docs/en/VISION.md) §8 uses quarterly **P-numbering** (P1–P3)
spanning all three packages. They are two views of one plan — the table below
maps between them. **VISION §8 is the strategic source of truth; the sequenced,
task-level execution lives in [DEV-PLAN.md](DEV-PLAN.md).**

| M (this file) | P (VISION §8) | Note |
|---|---|---|
| M0 Skeleton ✅ · M1 MVP ✅ | — (pre-P foundation) | Done before the P-framing; the numpy-only core SDK. |
| M2 Real Perception + Scale + Verbalizer 🟡 | **P1 · 地基** | P1 = close the M2 demo *without a GPU* + add the evidence layer the M-view didn't name (eval set v0, retrieval-context fix, perf gate, deploy-matrix smoke, dual Reasoner). |
| M3 Real Robot Demo + ROS 2 ⬜ | **P2 · 集成验证** | External open-friendly bodies + LeRobot + the RoboOS/InternRobotics spike outcome. |
| M4 Hardening + Mem0 Adapter ⬜ | **P3 · 生态位锁定** | Stable serialization protocol + default adoption + commercial baseline (OQ-1/OQ-2). |

## M0 · Skeleton ✅

- ✅ Repo, license (Apache-2.0), CI matrix (3.10–3.12 × mac/linux), package structure
- ✅ `Detection` / `Observation` / `Node` / `Edge` value objects + JSON round-trip
- ✅ SQLite schema + forward-only migrations (sqlite-vec wiring moved to M2 — see note)
- ✅ `pip install tempomem` works on Mac (no CUDA), no real perception
- ✅ `examples/01_quickstart.py` — fake detections in, query out

**Exit met:** `pytest -q` green; `import tempomem` on a clean Python 3.11 venv works.

> Deviation: M0 stored vectors as BLOB (numpy-only); sqlite-vec ANN landed in M2 (V1) behind the `[vec]` extra with a linear fallback. Logged in [05-OPEN](05-OPEN.md).

## M1 · MVP · "Detections-In" SDK ✅ (`v0.1.0a1`)

- ✅ `add_detections([Detection(...)])` ingest (BYO perception)
- ✅ Fusion arbiter v1: candidate search + 3D IoU + feature-cos + label scoring, deterministic merge/new/reject — incremental dedup (the value ConceptGraphs' single-shot pipeline lacks)
- ✅ `query(...)` → spatial + temporal + keyword retrieval, returns nodes
- ✅ **Semantic query** (pulled forward from M2): BYO `Encoder` protocol + `OpenClipEncoder` (`[clip]` extra), cosine over node features
- ✅ `serialize(format="prompt"|"json")` → compact graph text
- ✅ Web viewer (read-only) — `tempomem viz` exports a self-contained HTML scene (built in M2)
- ⬜ Replica/ScanNet RGB-D demo — needs the M2 perception adapter (CUDA)

**Exit:** README quickstart runs verbatim on a clean machine (`examples/01` + `02`). Replica demo rolls into M2.

## M2 · Real Perception + Scale + Verbalizer 🟡

Done (no GPU needed):
- ✅ V1 sqlite-vec ANN index (`[vec]`), maintained on write, linear fallback
- ✅ V2 `decay(half_life_days, min_conf)` + `forget()` — memory hygiene
- ✅ V3 LLM verbalizer: `Verbalizer` protocol + `answer()` (BYO OpenAI/Anthropic/Ollama)
- ✅ V4 split detection — `resplit()` (deterministic 2-means over member observations)
- ✅ V5 eval harness — `bench.recall_at_k`
- ✅ V6 `[clip]` + `[vec]` CI lanes
- ✅ P0/P2 `PerceptionAdapter` protocol + `add_frame(rgb, depth, pose)` seam (stub-tested)

Memory-deepening + retrieval (shipped after the V-track, no GPU needed):
- ✅ Hierarchy / rooms — `define_region(...)` + `contents(region)` (objects nest under regions)
- ✅ Spatial relations — `relate()` infers `near`/`on`/`under` edges + `related(node, rel=)`
- ✅ In-place correction — `update(node_id, ...)` + `history(node_id)` observation trail
- ✅ Relation-aware `serialize(format="prompt")` — appends each node's relation edges
- ✅ Multi-session merge — `merge(other_smem)` folds another store's objects via fusion
- ✅ Relational NL query — `query("what's on the table")` traverses relation edges
- ✅ Change detection — `moved()` / `changes(since_ts)` / `stale(before_ts)`
- ✅ Token-budgeted `serialize(format="prompt", max_tokens=N)` — bounded LLM payload
- ✅ `consolidate()` + `salient(n)` — merge missed duplicates, rank by recency·conf·evidence
- ✅ Dataset streaming — `DatasetSource` + `stream(mem, source)` + `SyntheticScene` (+ `bench.recall_at_k`)

Blocked on a CUDA dev box:
- ⛔ P1 concrete `ConceptGraphsAdapter` (SAM + Grounding DINO + OpenCLIP) behind `[perception]`
- ⛔ P3 bench parity on ConceptGraphs demo scenes (±10% object recall)
- ⛔ Recorded demo: "stream Replica scene, ask 5 questions, get 4 right"

**Exit (unchanged):** the recorded Replica demo. The protocol seam + wiring are done, so P1 is a drop-in once on a GPU.

## M3 · Real Robot Demo + ROS 2 Adapter ⬜

- ROS 2 bridge node (subscribe RGB-D topics, publish `/tempomem/scene_graph`)
- Public demo: mobile robot or AR session, multi-day persistence
- Benchmark vs eMEM on a shared dataset (open-vocab queries) — reuses `bench.recall_at_k`
- First external integration writeup (target: a Brain2Robot / L3-planner reference loop)
- v0.1.0 PyPI release + launch post
- 3D web viewer (the M2 `tempomem viz` is a 2D top-down read-only start)

**Exit:** 100 GitHub ★, 3 external users in Discord, 1 cited integration.

## M4 · Hardening + Mem0 Adapter ⬜ (Q3)

- gRPC server façade for multi-process / language-agnostic use
- Mem0 adapter shim (`Mem0SpatialBackend`)
- Vision Pro / Quest scene-mesh ingest adapter (sketch)
- nvblox (Apache-2.0) geometry-substrate adapter — optional, from the NVIDIA survey
- Hosted-tier prototype (managed `.smem` storage) — only if community pull

**Exit:** v0.2.0 with at least one production user.

## Cuts (deliberately not in 12-month plan)

- Training our own VLM or SLAM
- Closed-source cloud-only tier
- Action / planning layer (lives in consumers like the L3 planner)
- Multi-agent / shared map federation (interesting later, distraction now)

## Tracking

Each milestone breaks down in `docs/sprint/SPRINT-NN.md`. The sequenced
execution plan (GPU-aware, GT-adapter-first) lives in [DEV-PLAN.md](DEV-PLAN.md).
Resolved design questions land in [05-OPEN.md](05-OPEN.md); shipped surface is
tracked in [CHANGELOG.md](../../CHANGELOG.md) and [spec/API.md](../../spec/API.md).
