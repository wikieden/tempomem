# 03 · Roadmap

Milestones are time-boxed, not feature-locked. If a milestone slips on scope, cut scope, don't slip time.

**Status (2026-05-29):** M0 ✅ · M1 ✅ (`v0.1.0a1`) · M2 🟡 in progress (V-track + perception seam done; concrete RGB-D adapter + recorded demo blocked on a CUDA box). 49 tests, core install numpy-only. Per-milestone sprint breakdowns: [SPRINT-00](sprint/SPRINT-00.md) · [SPRINT-01](sprint/SPRINT-01.md) · [SPRINT-02](sprint/SPRINT-02.md).

Legend: ✅ done · 🟡 partial · ⛔ blocked · ⬜ not started.

## M0 · Skeleton ✅

- ✅ Repo, license (Apache-2.0), CI matrix (3.10–3.12 × mac/linux), package structure
- ✅ `Detection` / `Observation` / `Node` / `Edge` value objects + JSON round-trip
- ✅ SQLite schema + forward-only migrations (sqlite-vec wiring moved to M2 — see note)
- ✅ `pip install spatialmem` works on Mac (no CUDA), no real perception
- ✅ `examples/01_quickstart.py` — fake detections in, query out

**Exit met:** `pytest -q` green; `import spatialmem` on a clean Python 3.11 venv works.

> Deviation: M0 stored vectors as BLOB (numpy-only); sqlite-vec ANN landed in M2 (V1) behind the `[vec]` extra with a linear fallback. Logged in [05-OPEN](05-OPEN.md).

## M1 · MVP · "Detections-In" SDK ✅ (`v0.1.0a1`)

- ✅ `add_detections([Detection(...)])` ingest (BYO perception)
- ✅ Fusion arbiter v1: candidate search + 3D IoU + feature-cos + label scoring, deterministic merge/new/reject — incremental dedup (the value ConceptGraphs' single-shot pipeline lacks)
- ✅ `query(...)` → spatial + temporal + keyword retrieval, returns nodes
- ✅ **Semantic query** (pulled forward from M2): BYO `Encoder` protocol + `OpenClipEncoder` (`[clip]` extra), cosine over node features
- ✅ `serialize(format="prompt"|"json")` → compact graph text
- ✅ Web viewer (read-only) — `spatialmem viz` exports a self-contained HTML scene (built in M2)
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

Blocked on a CUDA dev box:
- ⛔ P1 concrete `ConceptGraphsAdapter` (SAM + Grounding DINO + OpenCLIP) behind `[perception]`
- ⛔ P3 bench parity on ConceptGraphs demo scenes (±10% object recall)
- ⛔ Recorded demo: "stream Replica scene, ask 5 questions, get 4 right"

**Exit (unchanged):** the recorded Replica demo. The protocol seam + wiring are done, so P1 is a drop-in once on a GPU.

## M3 · Real Robot Demo + ROS 2 Adapter ⬜

- ROS 2 bridge node (subscribe RGB-D topics, publish `/spatialmem/scene_graph`)
- Public demo: mobile robot or AR session, multi-day persistence
- Benchmark vs eMEM on a shared dataset (open-vocab queries) — reuses `bench.recall_at_k`
- First external integration writeup (target: a Brain2Robot / L3-planner reference loop)
- v0.1.0 PyPI release + launch post
- 3D web viewer (the M2 `spatialmem viz` is a 2D top-down read-only start)

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
tracked in [CHANGELOG.md](../CHANGELOG.md) and [spec/API.md](../spec/API.md).
