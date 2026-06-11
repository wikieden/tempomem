> 🌐 **English** · [中文](../zh/DEV-PLAN.md)

# Development Plan — 统一执行追踪（系统级）

This is the **single execution tracker** for the SpatialRobot workspace (three
packages: `spatialmem` / `spatialmem-perception` / `spatialmem-brain`).

- **Strategic anchor (WHY / quarterly WHAT):** [docs/VISION.md](../../../docs/en/VISION.md)
  §8 (P1/P2/P3). That file is authoritative for positioning, wedge, risks, GTM.
- **Milestone view (single-package M-numbering):** [03-ROADMAP.md](03-ROADMAP.md)
  (M0–M4) + its M↔P mapping table.
- **This file (WHAT next, in what order):** the concrete, sequenced task list
  that closes VISION §8 P1, given one hard constraint: **no GPU on the current
  dev box.**

> **Reconciliation note (2026-06-08).** This file previously disagreed with the
> system vision on the next step (it said "B1'/B5/A1 first"; the vision says
> "eval evidence first"). The two were at different altitudes and the old file
> also described a pre-split architecture. Resolved here into **one hybrid
> order** (fix the verified defect → real-data stream → eval set → ship), hung
> under VISION §8 P1. The redundant `docs/00-SYSTEM-VISION.md` (an earlier
> subset of `docs/VISION.md`) was deleted in the same pass.

## Where we are (2026-06-09)

- `v0.1.0a1` tagged, repo public. **176 tests pass (2 skipped)** across the
  workspace, core install numpy-only, `pyright` clean on core (0 errors).
- **Architecture already split** (this is done, not future): perception lives in
  the companion `spatialmem-perception` (`BoxDetectorAdapter` + `Detector3D`
  seam + cam→world geometry + `ImageEncoder`); the brain lives in
  `spatialmem-brain` (`Brain` / `Reasoner` / `CosmosReasonVerbalizer`). Core
  keeps only the `PerceptionAdapter` / `Verbalizer` protocols.
- **M0 ✅ · M1 ✅ · M2 🟡** — memory-deepening + retrieval tracks complete; the
  M2 recorded demo and learned (GPU) perception are the remaining gaps.

## P1 progress (2026-06-09)

The hybrid critical path below is **done through #3, with #4 partially done**.
176 tests pass (2 skipped); ruff + pyright clean. An adversarial-review pass
(5 agents) over #1–#4 caught one HIGH (a relational-query regression — fixed),
one MEDIUM (top-k cap → default `k=64`), and several LOW, all addressed.

- **#1 Retrieval-context fix ✅** — `Brain.ask()` now `query()`s a relevant
  subgraph (hits + 1-hop relation neighbours + hierarchy ancestors) and
  serializes only that via the new `serialize(node_ids=...)`; whole-scene
  fallback when nothing matches.
- **#2 B1' `ReplicaAdapter` ✅** — pure-numpy `gt_detections_from_frame`
  (deproject GT masks + depth + pose → world detections), `ReplicaAdapter`
  (`DatasetSource`), and `ReplicaFileReader` (real scenes, `[replica]` extra,
  honestly flagged unvalidated against real data). Geometry unit-tested; review
  verdict **correct** (math verified to machine epsilon).
- **#3 Eval set v0 ✅** — `bench.persistence_after_reopen` (restart /
  cross-episode recall), `bench.decay_forget` (lifecycle counts), and
  `citation_compliance` (format + validity rate). Deterministic, network-free.
- **#4 B5 demo 🟡 / A1 publish ⬜** — `examples/04_replica_demo.py` runs the
  ReplicaAdapter pipeline end to end and renders viz HTML (synthetic
  Replica-shaped frames: 12 frames → 3 fused nodes, recall@5 = 1.00). **The
  real-Replica recording awaits a dataset variant with per-frame instance GT**
  (the Nice-SLAM RGB-D zip lacks masks; use a ConceptGraphs-rendered Replica or
  a Habitat render). A1 PyPI publish is still pending (your call).

## Scope discipline — we are a MEMORY system, not perception

SpatialMem stores and queries spatial memory. **Recognition is not our job** —
like Mem0 doesn't do speech-to-text, we don't do object detection. Input is
caller-supplied detections (BYO perception). Consequences:

- Concrete learned perception (ConceptGraphs: SAM 2 + Grounding DINO + OpenCLIP)
  is an **optional `[perception]` extra in the companion repo, GPU-gated, off the
  critical path** to a working demo. It is a *quality upgrade*, not the demo
  blocker. (This is how VISION §8 P1's "first ConceptGraphsAdapter" milestone
  reconciles with the no-GPU constraint: it is **P1 scope but ⛔ CUDA-blocked**,
  same status as M2's learned-perception row.)
- Datasets (`SyntheticScene`, a future `ReplicaAdapter`) are **test/benchmark
  fixtures**, not product features. They feed the pipeline to prove the memory
  mechanics and to benchmark.

## The key reframe — the M2 demo does NOT need a GPU

M2's exit is "stream a Replica scene, ask 5 questions, get 4 right." That does
**not** need ConceptGraphs (SAM + Grounding DINO + OpenCLIP → CUDA). RGB-D
datasets (Replica, ScanNet, ARKitScenes) ship **ground-truth instance
segmentation + labels + poses**. A `ReplicaAdapter` that *reads those
annotations* produces `Detection`s with **zero model inference, zero GPU**, and
exercises the entire pipeline — `add_frame` → fusion → query → `answer` → eval.
So: **GT-adapter first (no GPU), learned perception later (GPU).** This reframe
is exactly what dissolves the "is GPU perception P1 or deferred?" conflict.

## Unified next-step order (hybrid — the reconciled sequence)

Priority resolves the old conflict: **fix the verified defect → real-data stream
→ eval evidence → visibility.** All four are CPU-only and on the critical path;
GPU perception sits off to the side until hardware lands.

| # | Task | Package | GPU | VISION §8 P1 row | Why this order |
|---|---|---|---|---|---|
| **1 ✅** | **Retrieval-context fix** — `Brain.ask()` must `query()` a relevant subgraph *then* `serialize`, not dump the whole graph token-truncated | brain | no | "检索式上下文（先 query 过滤子图再 serialize）" / OQ-6 | Fixes a **verified code defect** (see below) that otherwise makes any large-scene eval number a lie. Gates eval validity → must come first. |
| **2 ✅** | **B1' `ReplicaAdapter`** — parse a Replica scene's GT instance masks + depth + trajectory into the existing `DatasetSource` shape | core (+ fixture) | no | (feeds eval + closes M2 demo) | One real-data stream serves both the eval set and the recorded demo. Unblocks both downstream items at once. |
| **3 ✅** | **Eval set v0 (automated, deterministic)** — extend beyond the existing `bench.recall_at_k`: add **`cited_node_ids` format-compliance rate**, cross-episode persistence, decay/forget correctness. **No human semantic labelling** (that's P2). | core | no | "自建空间记忆评测集 v0（自动化、确定性）" | The wedge's number support. VISION makes this P1's headline; it depends on #1 (valid retrieval) and is strongest with #2 (real data). |
| **4 🟡** | **B5 record demo + A1 publish** — viz HTML + asciinema of the stream loop; PyPI publish | core | no | (visibility) | Visibility *after* there's a defensible number behind it. A1 is **your action** (needs PyPI token, irreversible). |

**Off the critical path (GPU-gated, do when CUDA lands):**

| Task | Package | Status | VISION §8 P1 row |
|---|---|---|---|
| `ConceptGraphsAdapter` (SAM 2 + Grounding DINO + OpenCLIP) behind `[perception]` | perception | ⛔ CUDA | "首个 PerceptionAdapter 具体实现" |
| `Cosmos3PerceptionAdapter` (Cosmos 3 boxes + ego-pose → world `Detection`) | perception | ⛔ CUDA | (companion backlog) |
| P3 parity vs ConceptGraphs demo scenes (±10% recall) | perception | ⛔ CUDA | "性能 / 质量证明" |

**Also in VISION §8 P1, parallelizable, no GPU (fold in opportunistically):**

- **Protocol v2 lifecycle** — add `__enter__`/`__exit__` (or `close()`/`flush()`)
  to `PerceptionAdapter` to manage stateful encoder (CLIP) lifecycle (HIGH-3).
- **Perf gate** — add a "1000 obs / 100 nodes `commit()` latency" upper-bound
  acceptance gate alongside the existing "100 obs / 30 ms".
- **Deploy-matrix smoke** — (a) no-GPU numpy core-only path (BYO `Detection` →
  query) end-to-end; (b) +perception GPU path. Both must pass.
- **Dual Reasoner backend** — Cosmos-Reason2 (local RTX PRO 6000, OpenAI-compat
  `/v1`) **and** RoboBrain both drive `Brain.ask()` → `Answer(cited_node_ids)`.
- **[P2 gating spike] RoboOS / InternRobotics scene-graph capability survey ✅
  (concluded 2026-06-10)** — verdict: **complementary, not covered** (RoboOS's
  shipped memory is a volatile Redis hash, `FLUSHDB` on master start; the
  serious design, RoboOS-NeXT STEM, is paper-only with code unreleased). OQ-5
  closed with monitoring triggers — re-check after BAAI Conf 2026-06-12/13.
  Evidence + triggers:
  [roboos-robobrain-deep-dive-2026.md](../../../docs/en/research/roboos-robobrain-deep-dive-2026.md) §3.5.

### The verified defect behind step #1 — fixed 2026-06-09

> **Fixed** in #1 above; kept here as the record of what was wrong.

`spatialmem-brain/src/spatialmem_brain/brain.py:61` (pre-fix) — the docstring says
"Retrieve memory → reason → answer", but the body does **not** retrieve:

```python
def ask(self, question: str) -> Answer:
    """Retrieve memory → reason → answer."""
    context = self._mem.serialize(format="prompt", max_tokens=self._budget)
    return self._reasoner.reason(question, context)
```

It serializes the **whole** graph, token-truncated to `_budget`. In a large
scene the relevant objects can fall off the end of the truncation, so recall
silently degrades and is **not** a function of memory quality. Fix: `query()`
(or `related()`) a relevant subgraph from `question` first, serialize *that*,
and have the eval (#3) report both strategies — full-graph truncation vs
query-subgraph — so the improvement is measured, not assumed (VISION §2.3, OQ-6).

## Brain track — how we build the "embodied brain" (decided 2026-06-10)

Strategy: **build the brain *system*, not a brain *model*.** Brain-model
competition is a red ocean (RoboBrain / Gemini-ER / Cosmos); every production
stack — including BAAI's own — converges on "external scene-graph store +
serialize into prompt + stateless VLM", which is exactly our three-package
shape. Evidence:
[roboos-robobrain-deep-dive-2026.md](../../../docs/en/research/roboos-robobrain-deep-dive-2026.md).

| # | Task | Package | GPU | Rationale (evidence) |
|---|---|---|---|---|
| B-T1 | **Close the loop** — refactor Brain2Robot into `spatialmem-brain`: `Brain.ask()` grows from answer-only to propose-action → execute → write the outcome back into memory | brain | no | a brain without a loop is a QA bot; refactor target decided 2026-06-09 |
| B-T2 | **Dual-reasoner A/B** — second vLLM endpoint serving RoboBrain 2.5-8B vs Cosmos-Reason2-8B, scored on eval set #3 (makes the "Dual Reasoner backend" bullet above concrete) | brain | serving only | vendor benchmarks only compare Cosmos-Reason**1**; what matters is who reads *our* scene-graph serialization better |
| B-T3 | **Memory add-ons** — append-only event-log table (scene deltas); embodiment partition (robot state / skills / battery); serialize a context bundle (scene ⊕ recent action feedback ⊕ robot state); align relation vocabulary to `{on,in,left,right,front,back,near}` | core | no | STEM ablations: drop spatial memory → steps 11.6→58.1; drop embodiment memory → 0% SR; 30.2% of failures = memory-noise accumulation (the fusion arbiter's exact target) |
| B-T4 | **MCP-ify the memory** — expose `spatialmem` query/answer/commit as an MCP server | core | no | RoboOS issues #76/#73 are live unmet demand for an open scene-graph memory component; locks the niche |

Sequencing: after P1 exit (#4); B-T3 is CPU-only and can fold in
opportunistically alongside the parallelizable P1 items. Out of scope
(unchanged): model training, VLA, multi-robot scheduling, simulators.
Deployment unchanged: cloud-first (RTX PRO 6000) → on-device Thor when it fits —
which is why Cosmos-Reason2 stays the primary reasoner (first-party Thor
support; RoboBrain has none documented).

## M3 reach (mostly no GPU) — after P1 exit

| ID | Task |
|---|---|
| D1 | ROS 2 bridge node (subscribe RGB-D, publish `/spatialmem/scene_graph`) |
| D2 | eMEM benchmark harness on a shared open-vocab dataset (reuses `bench`) |
| D3 | Drop the alpha: cut `v0.1.0`, launch post, first external-integration writeup |
| D4 | 3D web viewer (upgrade the 2D `spatialmem viz`) |

## Sequencing

```
   ┌─ 1 retrieval-context fix (brain, CPU) ─┐
   │                                        ▼
   └────────────────────────► 3 eval set v0 (CPU) ──► 4 B5 record + A1 publish ──► M3 (D-track) ──► v0.1.0
   2 B1' ReplicaAdapter (CPU) ──────────────┘
                                            │
   GPU perception (ConceptGraphs) ──────────┘  (folds in as a quality upgrade when CUDA lands)
```

Steps 1 and 2 are independent and can run in parallel; both feed 3. 4 is
visibility after the number exists. GPU perception is never on the critical path
to the demo — it upgrades quality once hardware is available.

## P1 exit criterion (from VISION §8)

One **reproducible** sentence proving the wedge isn't a slide: "memory holds X
objects, survives restart, on a deterministic synthetic scene recall@k = R,
`cited_node_ids` format-compliance = F, 1000-obs `commit()` latency < T, pure-numpy
path smoke passes." **Note: recall@k + format-compliance (auto-measurable), NOT
human-labelled semantic accuracy** — that is deferred to P2.

## Redlines

License redlines (no exceptions) and engineering redlines (RFC-gated) are
normative in [VISION §9](../../../docs/en/VISION.md). The three engineering invariants
that bind this tracker: core stays **numpy-only**; **fuse-before-persist** (new
mutators call `_flush_pending()` first); tests **network-free**
(`ScriptedReasoner` / `HashEncoder`).
