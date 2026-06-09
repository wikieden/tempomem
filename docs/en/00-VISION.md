> 🌐 **English** · [中文](../zh/00-VISION.md)

# 00 · Vision

> 单包（记忆核）愿景。系统级（SpatialRobot 三层整体）愿景见 [../../../docs/en/VISION.md](../../../docs/en/VISION.md)。

## One-Liner

**SpatialMem is the persistent, queryable, LLM-native spatial memory layer that sits between a robot/AR/agent's perception stack and its language model — what Mem0 is for text, SpatialMem is for 3D space.**

## Problem

Today an embodied or AR agent has three disjoint pieces:

1. **Perception** (SLAM + VLM): produces transient point clouds, occupancy grids, per-frame detections.
2. **Reasoning** (LLM / VLA): consumes the *current* perception, picks an action.
3. **Long-term memory**: nothing. Or a homemade pickle of scene graphs that doesn't survive a restart, doesn't update incrementally, and can't be queried in natural language.

Every embodied team rebuilds the same plumbing — fuse detections, dedupe objects, persist, serialize for the LLM. Each does it poorly because it's not their core IP.

## What We Build

A **framework-agnostic spatial memory SDK** with three contracts:

- **Ingest contract** — `add_frame(rgb, depth, pose)` or `add_pointcloud(...)` or `add_detections(...)`. Swappable perception backends (start with ConceptGraphs-style RGB-D fusion; later: Hydra, nvblox-features, custom).
- **Memory contract** — incremental scene graph (objects, places, rooms, relations) persisted to a single-file store. Survives restart. Supports update / forget / decay.
- **Query contract** — `mem.query("where is the red mug?")` returns grounded answer + node ids; `mem.serialize()` exports compressed graph text for any LLM prompt; `mem.spatial(near=..., radius=...)` for geometric filters.

## Non-Goals

- **Not a new SLAM.** Pose comes from upstream (ORB-SLAM3, cuVSLAM, ARKit, sim ground truth).
- **Not a new VLM.** Embeddings come from upstream (CLIP / SigLIP / user model).
- **Not a robot OS.** No motion planning, no ROS hard-dependency. ROS 2 node provided as adapter, not as core.
- **Not a cloud service first.** Local-first, embedded, BYO-LLM. Hosted tier comes later as convenience, not moat.

## Wedge

> Mem0 is text memory at \$24M Series A and 56k★. Spatial memory is the same shape of problem, on a bigger substrate, and currently *empty*.

The closest thing shipping is `eMEM` (ROS 2-locked, 2★, no open-vocab semantics). Academic repos (ConceptGraphs, Hydra, Open3DSG, DovSG) all stop at "graph built" — none productize persistence, query, or LLM serialization.

The L3 planner / Brain2Robot work in adjacent repos is a built-in flagship consumer — we ship the memory layer the planner needs, with a real demo of "memory → planning" no academic repo can match.

## Audience (priority order)

1. **Embodied agent researchers** wanting a drop-in spatial memory so they don't reimplement object dedup + persistence. Win them with `pip install` + 10-line demo.
2. **Humanoid / mobile-robot startups** doing multi-day autonomy. Win them with ROS 2 adapter + benchmark on persistence under drift.
3. **AR/XR app developers** (Vision Pro, Quest) needing scene memory across sessions. Win them with iOS/Android bridge later.
4. **Smart-home / facility robots.** Same shape, later.

## Bet

Two 2026 narratives — "agent memory" (Mem0 thesis) and "Physical AI" (NVIDIA thesis) — are converging. SpatialMem is the only library positioned to be cited in both stories. We win if we ship the cleanest API, not the deepest research.

NVIDIA's June-2026 **Cosmos 3** technical report makes the bet for us: it names *"temporally persistent state, spatial grounding tied to objects and agents … a maintained, actionable scene estimate"* as the unsolved challenge for Physical AI — then ships a per-clip model with a bounded context and **no** persistent scene store. The biggest player in Physical AI just defined our layer as the missing piece. Cosmos 3's Reasoner (structured camera-frame 3D boxes + metric ego-pose) becomes an *upstream feed* for our ingest, not a competitor — it perceives, we remember.

## Success in 12 months

- 1k★ on GitHub
- 5 real-world integrations cited by name (1 humanoid startup, 1 AR app, 3 research labs)
- Quoted alongside Mem0 in at least one "agent memory" survey
- Hostable as a paid service for teams that don't want to run the vector store
