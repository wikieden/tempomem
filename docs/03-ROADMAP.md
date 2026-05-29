# 03 · Roadmap

Milestones are time-boxed, not feature-locked. If a milestone slips on scope, cut scope, don't slip time.

## M0 · Skeleton (Week 1–2)

- Repo, license, CI, package structure
- `Frame` / `Observation` / `Node` / `Edge` dataclasses + JSON round-trip
- SQLite schema + migrations + sqlite-vec wiring
- `pip install spatialmem` works on Mac (no CUDA), no real perception yet
- 50-line "fake detections in, query out" notebook demo

**Exit:** `pytest -q` green; `import spatialmem` on a clean Python 3.11 venv works.

## M1 · MVP · "Detections-In" SDK (Week 3–5)

- `add_detections([Detection(...)])` ingest path (BYO perception — user runs CLIP+SAM themselves)
- Fusion arbiter v1: KNN + 3D IoU + CLIP-cos scoring, deterministic merge
- `query("where is X?")` → spatial + semantic retrieval, returns nodes
- `serialize(format="prompt")` → compact graph text
- One demo: Replica/ScanNet RGB-D dataset → graph → query
- Web viewer (read-only)

**Exit:** README quickstart runs verbatim on a clean machine in <5 minutes.

## M2 · ConceptGraphs Adapter (Week 6–8)

- `add_frame(rgb, depth, pose)` via the ConceptGraphs backend (pinned commit, packaged extras)
- Bench on ConceptGraphs' own demo scenes; match their object recall ±10%
- Incremental ingest (their reference impl is single-shot — this is new value)
- Decay / forget API
- LLM verbalizer (BYO model — OpenAI / Anthropic / Ollama)

**Exit:** "Stream Replica scene, ask 5 questions, get 4 right" demo recorded.

## M3 · Real Robot Demo + ROS 2 Adapter (Week 9–12)

- ROS 2 bridge node (subscribe RGB-D topics, publish `/spatialmem/scene_graph`)
- Public demo: mobile robot or AR session, multi-day persistence
- Benchmark vs eMEM on a shared dataset (open-vocab queries)
- First external integration writeup (target: a Brain2Robot / L3-planner reference loop)
- v0.1.0 PyPI release + launch post

**Exit:** 100 GitHub ★, 3 external users in Discord, 1 cited integration.

## M4 · Hardening + Mem0 Adapter (Q3)

- gRPC server façade for multi-process / language-agnostic use
- Mem0 adapter shim (`Mem0SpatialBackend`)
- Vision Pro / Quest scene-mesh ingest adapter (sketch)
- Hosted-tier prototype (managed `.smem` file storage) — optional, only if community pull

**Exit:** v0.2.0 with at least one production user.

## Cuts (deliberately not in 12-month plan)

- Training our own VLM or SLAM
- Closed-source cloud-only tier
- Action / planning layer (lives in consumers like the L3 planner)
- Multi-agent / shared map federation (interesting later, distraction now)

## Tracking

Every milestone breaks down in `docs/sprint/SPRINT-NN.md` once started, mirroring the clihub `14-SPRINT.md` format.
