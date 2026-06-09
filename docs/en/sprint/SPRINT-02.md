> 🌐 **English** · [中文](../../zh/sprint/SPRINT-02.md)

# Sprint 02 · Real Perception + Scale + Verbalizer (M2)

**Goal:** Close the loop from raw RGB-D to answered question. Today SpatialMem is detections-in (BYO perception). M2 ships a real `add_frame(rgb, depth, pose)` perception adapter so a user streams a dataset scene and asks questions with no hand-fed detections. Plus the pieces that make it production-shaped: ANN retrieval at scale, decay/forget, and an LLM verbalizer.

**Exit (from [roadmap](../03-ROADMAP.md)):** "Stream Replica scene, ask 5 questions, get 4 right" demo recorded.

## DECIDED 2026-05-29 — perception backend = Option C (Protocol + ConceptGraphs first)

NVIDIA open-model research done (see [05-OPEN.md](../05-OPEN.md) P1/P3 log). Verdict: **no NVIDIA model fits as a bundled default** — FoundationStereo/Pose/RADIO are non-commercial (NSCL), Isaac Perceptor is proprietary, NV-CLIP is enterprise-only, C-RADIO is commercial-OK but has no native text encoder (breaks our text-query ANN). So: **Option C** — define `PerceptionAdapter` Protocol (P0), ship **ConceptGraphs (SAM + Grounding DINO + OpenCLIP, Apache/MIT)** as the first concrete adapter, pinned + soft-forked. NVIDIA pieces revisited later: nvblox geometry (M3/M4), C-RADIO `[radio]` encoder (post-M2, needs text-space prototype), Cosmos-Reason2 (named BYO verbalizer in docs). **P-tasks unblocked.**

`add_frame` needs a backend that turns RGB-D → open-vocab 3D detections (label + 3D bbox + feature). Options considered:

| Option | Backend | License | Install burden | Notes |
|---|---|---|---|---|
| **A** | ConceptGraphs (SAM + open-CLIP), pinned commit | Apache/MIT-ish | conda + torch + faiss | academic, single-shot; our incremental fusion is the new value |
| **B** | NVIDIA stack (nvblox geometry + RADIO/NV-CLIP features + open-vocab seg) | ⚠️ many NVIDIA model licenses are non-commercial / eval-only | CUDA-only | research deciding fit + license traps |
| **C** | Hybrid: our adapter Protocol, ship ConceptGraphs first, NVIDIA as optional extra | per-component | mixed | most flexible, most work |

→ **Decided: C** (above). `PerceptionAdapter` Protocol + ConceptGraphs first.

## Task Breakdown

### Not gated — start immediately

| ID | Task | Output | Est (CC) |
|---|---|---|---|
| V1 | sqlite-vec wiring: `obs_features`/`node_features` vec0 tables behind `[vec]` extra; migration 002; ANN path in `query.semantic_vec` with linear fallback when extension absent | ANN at >10k nodes | 60 min |
| V2 | `decay(half_life_days, min_conf)` — confidence decay by age, prune below floor; `forget()` already exists | memory hygiene API | 40 min |
| V3 | LLM verbalizer: `Verbalizer` Protocol + `answer(query, k)` = retrieve → serialize(prompt) → BYO LLM (OpenAI/Anthropic/Ollama); no bundled key | NL answers, not just nodes | 50 min |
| V4 | Split detection: one node drifts into two clusters → split (deferred from M1) | fusion correctness | 45 min |
| V5 | Eval harness: scene → ingest → N scripted queries → recall@k vs ground truth; reusable for the demo metric | bench number | 50 min |
| V6 | `[clip]` CI job: install extra, validate `OpenClipEncoder` text embed shape/dim against real torch (built in M1, untested in CI) | green clip lane | 25 min |

### Gated on perception-backend decision

| ID | Task | Output | Depends |
|---|---|---|---|
| P0 | `PerceptionAdapter` Protocol: `process_frame(rgb, depth, pose) -> list[Detection]` | backend-agnostic seam | — |
| P1 | First concrete adapter (A/B/C winner), pinned commit, packaged extra | `add_frame` works | P0 + decision |
| P2 | `SpatialMemory.add_frame(rgb, depth, pose)` wired through adapter → fusion | RGB-D ingest | P1 |
| P3 | Bench adapter on backend's own demo scenes; object recall within ±10% of reference | parity proof | P1, V5 |

## Out of scope (M3+)

- ROS 2 bridge → M3
- Real robot / AR multi-day persistence → M3
- eMEM benchmark → M3
- gRPC façade, Mem0 adapter → M4

## Definition of Done

- [x] `PerceptionAdapter` Protocol + `add_frame` ingests RGB-D **(P0/P2 — stub-tested)**; concrete ConceptGraphs adapter (P1) pending CUDA dev box
- [x] split detection — `resplit()` **(V4 — built 2026-05-29)**
- [x] sqlite-vec ANN path green (with linear fallback); `[vec]` extra installs **(V1 — built 2026-05-29)**
- [x] `decay()` + `answer()` (verbalizer) APIs land with tests **(V2, V3 — built 2026-05-29)**
- [x] Eval harness reports recall@k **(V5 — `spatialmem.bench.recall_at_k`)**
- [x] `[clip]` CI lane green **(V6 — smoke test, no weight download)**
- [ ] Demo recorded: stream Replica scene → 5 questions → ≥4 right
- [ ] Coverage ≥ 80% on new modules; ruff clean

## Risks

| Risk | Mitigation |
|---|---|
| NVIDIA model licenses non-commercial → can't ship as default | Protocol seam (P0) keeps backend swappable; default to Apache-compatible ConceptGraphs, NVIDIA as opt-in extra |
| Perception backend drags conda/CUDA into install | Isolate in `[perception]`/`[nvidia]` extras; core stays numpy-only |
| sqlite-vec wheel gaps across OS/Py | Linear-cosine fallback already exists; ANN is opportunistic, not required |
| Verbalizer needs a key/network | BYO model only; no bundled dependency, offline path via Ollama |
