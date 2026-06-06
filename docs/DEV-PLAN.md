# Development Plan

Concrete, sequenced execution plan layered on top of [03-ROADMAP.md](03-ROADMAP.md).
The roadmap says *what milestones*; this says *what to build next and in what
order*, given one hard constraint: **no GPU on the current dev box.**

**Where we are (2026-06-05):** `v0.1.0a1` tagged, repo public, 52 tests / 95%
cov, CI green; the memory-deepening + retrieval tracks are complete. M0 ✅ ·
M1 ✅ · M2 🟡 (V-track + perception seam + viz done; learned perception P1/P3
blocked on CUDA).

## Scope correction (2026-06-05) — we are a MEMORY system, not perception

SpatialMem stores and queries spatial memory. **Recognition is not our job** —
like Mem0 doesn't do speech-to-text, we don't do object detection. Input is
caller-supplied detections (BYO perception). Consequences:

- **Perception (Phase C / ConceptGraphs) is demoted to an optional `[perception]`
  extra + a "how to wire your own perception" example.** Not a core deliverable.
  Off the critical path entirely.
- **Datasets (`SyntheticScene`, a future `ReplicaAdapter`) are test/benchmark
  fixtures, not product features.** They feed the pipeline to prove the memory
  mechanics (and later to benchmark vs eMEM). Not a headline capability.

The real core track is **deepening the memory**, all perception-free —
**shipped 2026-06-05** (`define_region`/`contents`, `relate`/`related`,
`update`/`history`, relation-aware `serialize`, `merge`):

| Track | What it unlocks | Status |
|---|---|---|
| Hierarchy / rooms | object → region → room; "what's in the kitchen?" | ✅ `define_region` / `contents` |
| Relations / edges | "mug *on* counter", "chair *near* table"; "what's on the table?" | ✅ `relate` / `related` |
| update / history | correct a memory; "the mug moved / was last seen…" | ✅ `update` / `history` |
| relation-aware serialize | LLM prompt carries structure, not just an object list | ✅ `serialize(relations=True)` |
| multi-session merge | re-enter a space, memory continues without re-creating nodes | ✅ `merge` |

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

## Phase C — Learned perception → companion repo `spatialmem-perception`

**Scope decision (2026-06-06):** SpatialMem core does **not** do perception. It
ships only the BYO seam — the `PerceptionAdapter` protocol (`add_frame`) — and
stays numpy-only. All concrete perception (open-vocab detectors, depth/pose
camera→world lift, image-crop feature encoding) lives in a **separate companion
project, `spatialmem-perception`**, which depends on `spatialmem`. Perception is
a peripheral, not a core deliverable.

Companion-repo backlog (needs GPU; tracked in `spatialmem-perception`, not here):

| ID | Task | Output |
|---|---|---|
| C1 | `ConceptGraphsAdapter` (SAM + Grounding DINO + OpenCLIP) | open-vocab perception |
| C2 | `Cosmos3PerceptionAdapter` — Cosmos 3 Reasoner 3D boxes + ego-pose → world `Detection` + CLIP-crop feature | Cosmos-native perception |
| C3 | parity: detector recall vs GT, within ±10% | quality proof |
| C4 | record the demo with learned perception | "real" demo |

The companion repo owns the geometry cam→world lift, the `ImageEncoder` seam +
CLIP image encoder, and every concrete adapter. Core keeps only the protocol.

> Path-A `CosmosReasonVerbalizer` (the Cosmos **answer** brain) shipped in core —
> see `spatialmem.cosmos`; it is an LLM verbalizer, **not** perception. Full
> system design (Cosmos 3 + SpatialMem + LLM "brain", contracts/topologies/eval):
> [design/cosmos3-spatialmem-llm-brain.md](design/cosmos3-spatialmem-llm-brain.md).

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

1. **B1'** — real `ReplicaAdapter` over a downloaded Replica scene (GT masks +
   depth + trajectory → the same detection stream `SyntheticScene` already
   feeds). `DatasetSource` / `SyntheticScene` / `stream` (B1) are done; this
   swaps the synthetic source for a real parser and is the demo unblock.
2. **B5** — record the demo artifact (viz HTML of the final graph + asciinema
   of the stream loop) once B1' lands.
3. **A1** — publish to PyPI (your call; I'll prep the exact commands + TestPyPI dry-run).

Open question for you: which Phase-B dataset — **Replica** (clean synthetic,
small, easy license) or **ScanNet** (real scans, heavier, registration needed)?
Replica is the faster path to the demo.
