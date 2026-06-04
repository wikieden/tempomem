# Development Plan

Concrete, sequenced execution plan layered on top of [03-ROADMAP.md](03-ROADMAP.md).
The roadmap says *what milestones*; this says *what to build next and in what
order*, given one hard constraint: **no GPU on the current dev box.**

**Where we are (2026-06-04):** `v0.1.0a1` tagged, repo public, 52 tests / 95%
cov, CI green. M0 ✅ · M1 ✅ · M2 🟡 (V-track + perception seam + viz done;
learned perception P1/P3 blocked on CUDA).

## The key reframe — the M2 demo does NOT need a GPU

M2's exit is "stream a Replica scene, ask 5 questions, get 4 right." We assumed
that needs ConceptGraphs (SAM + Grounding DINO + OpenCLIP → CUDA). It doesn't.

RGB-D datasets (Replica, ScanNet, ARKitScenes) ship **ground-truth instance
segmentation + labels + poses**. A `DatasetAdapter` that *reads those
annotations* produces `Detection`s with **zero model inference, zero GPU**. That
exercises the entire pipeline — `add_frame` → fusion → query → `answer` → eval —
and gets the recorded demo. Learned perception (ConceptGraphs from raw pixels)
becomes a *quality upgrade*, not the demo blocker.

So: **GT-adapter first (no GPU), learned perception later (GPU).**

## Phase A — Ship & visibility (no GPU, ~half day)

Make the thing installable and legible before adding features.

| ID | Task | Output |
|---|---|---|
| A1 | Publish `v0.1.0a1` to PyPI (sdist+wheel built, `twine check` passed) — TestPyPI dry-run, then real. **User action** (needs PyPI token; irreversible). | `pip install spatialmem` works |
| A2 | README badges: CI status, PyPI version, license, Python versions | credible landing |
| A3 | GitHub issue + PR templates (`.github/ISSUE_TEMPLATE/`, `PULL_REQUEST_TEMPLATE.md`) | contribution on-ramp |
| A4 | Capture a viz screenshot / short asciinema of the quickstart for the README | shows it's real |

## Phase B — Close the M2 demo WITHOUT a GPU (~2–3 days CC)

The unblock. Ground-truth perception adapter → end-to-end demo.

| ID | Task | Output | Depends | Status |
|---|---|---|---|---|
| B1 | `DatasetSource` protocol + `SyntheticScene` (deterministic GT stream) + `HashEncoder` fixture | GT detections, no GPU | — | ✅ |
| B2 | `stream(mem, source, commit_every=...)` ingest helper | scene → graph | B1 | ✅ |
| B3 | `examples/03_stream_scene.py` (synthetic fixture, CI-runnable) | runnable demo | B2 | ✅ |
| B4 | Eval: scripted Q/A, `bench.recall_at_k` — demo gets **5/5**, test asserts ≥0.8 | the M2 exit metric | B3 | ✅ |
| B1' | Real `ReplicaAdapter`: parse a Replica scene's GT instance masks + depth + trajectory into the same `DatasetSource` shape | real-data stream | B1 | ⬜ |
| B5 | Record: viz HTML of the final graph + asciinema of the stream loop | demo artifact | B4 | ⬜ |

**Done (2026-06-04):** the synthetic path proves the pipeline end-to-end on CPU
— 15 frames × 5 objects = 75 observations fuse to 5 nodes (15× dedup), recall
5/5. Remaining: swap the synthetic source for a real Replica parser (B1') and
record the artifact (B5). P3 parity vs ConceptGraphs waits on a GPU.

## Phase C — Learned perception (needs GPU, when available)

Runs on Colab / a rented cloud GPU — parallel to B/D, not blocking.

| ID | Task | Output |
|---|---|---|
| C1 | `ConceptGraphsAdapter` (SAM + Grounding DINO + OpenCLIP) behind `[perception]`, same `PerceptionAdapter` seam | real open-vocab perception |
| C2 | P3 parity: object recall of C1 vs B1 GT on shared scenes, within ±10% | quality proof |
| C3 | Swap demo from GT to learned perception; re-record | "real" demo |

## Phase D — M3 reach (mostly no GPU)

| ID | Task |
|---|---|
| D1 | ROS 2 bridge node (subscribe RGB-D, publish `/spatialmem/scene_graph`) |
| D2 | eMEM benchmark harness on a shared open-vocab dataset (reuses `bench`) |
| D3 | Drop the alpha: cut `v0.1.0`, launch post, first external-integration writeup |
| D4 | 3D web viewer (upgrade the 2D `spatialmem viz`) |

## Sequencing

```
A (ship) ──┐
           ├──> B (GT demo, CPU) ──> D (M3) ──> v0.1.0
C (GPU) ───┘        (C folds into D's demo when GPU lands)
```

A and B are the immediate path — both CPU-only, both high-leverage. C waits for
hardware and is not on the critical path to a working demo. Do A1 (publish)
first so the badges + install line in A2 are real.

## Immediate next 3

1. **A1** — publish to PyPI (your call; I'll prep the exact commands + TestPyPI dry-run).
2. **A2/A3** — badges + issue/PR templates (I can do now).
3. **B1** — start the `ReplicaAdapter` (the demo unblock).

Open question for you: which Phase-B dataset — **Replica** (clean synthetic,
small, easy license) or **ScanNet** (real scans, heavier, registration needed)?
Replica is the faster path to the demo.
