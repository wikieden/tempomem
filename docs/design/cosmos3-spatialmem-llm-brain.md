# Design — Cosmos 3 + SpatialMem + LLM as an embodied "brain"

**Status:** design / forward-looking (v2, post adversarial review).
**Audience:** integrators wiring SpatialMem into an embodied or video-understanding agent.
**Companion docs:** [../02-ARCHITECTURE.md](../02-ARCHITECTURE.md) (SpatialMem
internals), [../01-POSITIONING.md](../01-POSITIONING.md) (why the memory layer
is the missing piece). Concrete perception (the `Cosmos3PerceptionAdapter`, the
camera→world geometry lift, image-crop encoding) lives in the separate companion
repo **`spatialmem-perception`** — core does not do perception.

> Reviewed by an adversarial critic panel (architecture / API-accuracy /
> completeness). Corrections folded: API signatures aligned to shipped code,
> concurrency claims made honest, security + store-compat + C4 + error model
> added. One real code bug it surfaced (maintenance `commit()` can flush
> un-fused pending observations) is tracked separately, not in this doc.

## 1. Thesis

An embodied agent needs three faculties that no single model provides:

1. **Perceive + physically reason** about what the camera sees, *now*.
2. **Remember** what it has seen, *persistently*, in a structured, queryable
   spatial form — across clips, sessions, days.
3. **Reason / plan / talk** in general language about goals and the remembered
   world.

NVIDIA **Cosmos 3** gives (1). **SpatialMem** gives (2). A general **LLM** gives
(3). Cosmos 3's own technical report names the (2) gap explicitly — *"temporally
persistent state, spatial grounding tied to objects and agents … a maintained,
actionable scene estimate"* — and ships none of it: the generator runs on a
bounded ~74K-token context with per-request KV-cache, no cross-session store.
This design wires the three into one cognitive loop.

### Brain analogy (informal, for intuition)

| Faculty | Component | Brain region |
|---|---|---|
| Vision + physical intuition | Cosmos 3 Reasoner | visual cortex + cerebellum |
| Persistent spatial/episodic memory | SpatialMem | hippocampus |
| Language, planning, abstraction | LLM | prefrontal cortex |

The "brain" is not any one box — it is the **loop** that binds them.

## 2. Conventions (one source of truth for all three buses)

| Quantity | Convention |
|---|---|
| World frame | right-handed, **meters** |
| Pose | 4×4 homogeneous **camera→world** |
| Positions in C2/C3 outputs | world-frame meters |
| Timestamps | **float epoch seconds** (matches `node.t_last`) |
| Orientation (if surfaced) | radians, roll/pitch/yaw |
| Node identity | integer `node_id`, stable within a read snapshot (§6) |

Cosmos sees **RGB only** at inference. Depth + pose are consumed by the
*adapter* and SpatialMem (the camera→world lift), **not** by the Cosmos
Reasoner.

## 3. Architecture

```
        ┌─────────────────────────── PERCEIVE ──────────────────────────┐
        │                                                                │
  RGB video ───► Cosmos 3 Reasoner ──┐   depth + pose (SLAM/ARKit/sim)   │
                 (RGB-only, System-2)│           │                       │
                                     ▼           ▼                       │
                          ┌───────────────────────────┐                 │
                          │  Cosmos3PerceptionAdapter  │                 │
                          │  JSON 3D box (cam frame) +  │                │
                          │  pose → world Detection +   │                │
                          │  CLIP-crop feature          │                │
                          └─────────────┬──────────────┘                 │
                                        ▼                                 │
                                 ┌────────────┐                          │
                                 │ SpatialMem │  fuse·persist·hierarchy  │
                                 │ (memory)   │  ·relations·temporal     │
                                 └─────┬──────┘                          │
                       serialize(prompt)│ + query/semantic/spatial tools │
                                        ▼                                 │
                                 ┌────────────┐                          │
                                 │    LLM     │  reason · plan · answer   │
                                 │  (cortex)  │ ◄─ C4 active perception ──┘
                                 └─────┬──────┘   ("look closer at node N")
                                       ▼
                          action / NL answer
                                       └── write-back: update()/history
```

Four contracts (§4), one shared memory, one tick (§5), one concurrency
contract (§6).

## 4. Interface contracts

### C1 · Cosmos 3 → SpatialMem (perception → memory) — *seam in core; adapter in `spatialmem-perception`*

Core ships only the `PerceptionAdapter` protocol (`add_frame`). The concrete
`Cosmos3PerceptionAdapter` lives in the companion repo **`spatialmem-perception`**:
Cosmos sees **RGB** and emits structured **camera-frame** 3D boxes (JSON:
`label, center, size, orientation`) and can *estimate* metric ego-pose; the
adapter lifts each box to a world-frame `Detection` (using the caller-supplied
pose where available, see §7) and attaches a CLIP-crop feature — Cosmos emits no
per-object embedding. The camera→world geometry lift and the image-crop encoder
also live in that companion repo.

```python
# shipped API (adapter is the WIP piece):
mem.add_frame(rgb, depth, pose, adapter=cosmos3_adapter)  # adapter.process_frame -> dets -> queue
mem.commit()                                              # arbiter dedups across frames
```

### C2 · SpatialMem → LLM (memory → cortex) — *shipped*

The serialized scene graph is the context bus: compact, hierarchical,
relation-aware, token-budgeted.

```python
scene = mem.serialize(format="prompt", max_tokens=1500)   # rooms > objects > relations
```

`max_tokens` is an **approximate** budget (~4 chars/token heuristic, not a real
tokenizer); it drops whole top-level subtrees most-recent-first with a
`(N more omitted)` marker (a region never appears without its contents). Leave
headroom against the model's real tokenizer.

### C3 · LLM ↔ SpatialMem (cortex queries memory as tools) — *shipped: `spatialmem.SpatialMemTools`*

C2 and C3 are two access patterns of **one** memory-access seam: hand the whole
serialized graph (C2) for small scenes, or let the LLM pull on demand (C3) for
large ones. Both read the **same per-turn snapshot** (§6) so cited `node_id`s
stay valid for the turn.

Proposed tool surface (thin wrapper; underlying methods all shipped):

| Tool | Maps to (shipped) | Status |
|---|---|---|
| `spatial_query(near, radius_m)` | `mem.spatial(...)` | shipped fn |
| `semantic_search(text, k)` | `mem.semantic(...)` | shipped fn |
| `whats_in(region)` | `mem.contents(region)` | shipped fn |
| `whats_on(anchor)` | `mem.query("what's on the …")` | shipped fn |
| `recent_changes(since_ts)` | `mem.changes(since_ts)` / `mem.moved()` | shipped fn |
| `serialize_scene(max_tokens)` | `mem.serialize(...)` | shipped fn |

Concrete tool schema (one example; others follow the same shape):

```json
{
  "name": "semantic_search",
  "description": "Find remembered objects matching a text query.",
  "parameters": {
    "type": "object",
    "properties": {
      "text":  {"type": "string", "description": "e.g. 'coffee mug'"},
      "k":     {"type": "integer", "default": 5}
    },
    "required": ["text"]
  }
}
```

Return envelope (every tool), so the LLM can **cite** ids (anti-hallucination):

```json
{"hits": [
  {"node_id": 42, "label": "mug", "centroid_m": [1.2, 0.3, 0.9],
   "confidence": 0.87, "t_last": 1.748e9}
]}
```

`near` accepts a `node_id`, a label, or an `[x,y,z]` (world meters); `radius_m`
is meters; `since_ts` is float epoch seconds.

### C4 · LLM → Cosmos 3 (active perception) — *proposed, bounded*

When memory is uncertain (low-confidence node, stale region) the LLM may request
a fresh Cosmos look. C4 is a **bounded** request, not an open loop:

- **Target representation:** `node_id` or world bbox → resolve to a *required
  camera pose*; if the target is not currently reachable (out of view, robot
  can't get there), return `target_unreachable` as a first-class result, do not
  block.
- **Budget:** max re-looks per tick and per node; exceeding it → answer with
  declared uncertainty.
- **Cycle-breaker:** a node that stays low-confidence after *k* re-looks enters a
  cooldown and is not re-requested until new evidence arrives.
- **Failure:** Cosmos timeout / malformed output → degrade to "reason over
  existing memory, flag staleness"; never stall the reason loop.

## 5. Deployment topologies

### Topology A — Cosmos Reasoner *is* the LLM (single-VLM brain)

Cosmos does perception AND final reasoning/answer; SpatialMem is the memory
between clips. All-NVIDIA, simplest wiring, GPU-heavy. Uses the shipped
`CosmosReasonVerbalizer` as the `answer()` backend — which **today defaults to
Cosmos Reason 2 (`nvidia/cosmos-reason2-8b`)** and is `model=`-configurable to
the Cosmos 3 Reasoner NIM once GA.

### Topology B — separate general LLM (recommended default)

Cosmos only perceives (C1). A BYO LLM (Claude/GPT/Llama/Ollama) reasons over the
serialized graph (C2) and calls memory tools (C3). More flexible, cheaper
reasoning, no lock-in; matches SpatialMem's BYO-`Verbalizer` design.

| | A (Cosmos = brain) | B (separate LLM) |
|---|---|---|
| Reasoning quality | physical-AI tuned | best general LLM |
| Cost / latency | GPU per reason step | cheap text LLM |
| Lock-in | NVIDIA | none (BYO; local Ollama possible) |
| Wiring | simplest | one more component |

## 6. Concurrency & consistency contract (v0 = single-threaded)

**Honest statement of what the code does today:** `SpatialMemory` wraps one
synchronous `sqlite3.Connection`. It is **not** thread-safe; callers **must
serialize all access**. `add_frame()` runs `adapter.process_frame()` **inline**
— it blocks the caller for Cosmos's seconds-per-clip latency. There is no worker
thread or async ingest queue yet.

Invariants:
- **No maintenance while observations are pending.** `add_detections()` defers
  fusion to `commit()`; do not call `decay()/consolidate()/relate()/update()/
  define_region()/forget()` between `add_detections()` and `commit()`. (A code
  fix to enforce this is tracked separately.)
- **Per-turn read snapshot.** A reasoning turn should read one snapshot for all
  its C2 serialize + C3 tool calls, so `node_id`s don't dangle if a perception
  commit lands mid-turn (a node could be merged/split/pruned by
  consolidate/decay between two tool calls).

**Future (not shipped):** a bounded ingest queue + dedicated writer connection +
WAL reader connections per query thread would let perception (sampled) and
reasoning (on-demand) truly run on independent cadences. Until then, the "slow
eye / on-demand cortex" decoupling is a *scheduling* discipline in the caller,
not a concurrency guarantee from the library.

## 7. The cognitive tick (sequence)

```
on new keyframe(s) + pose:
  1. PERCEIVE  dets = cosmos3_adapter.process_frame(rgb, depth, pose)  # System-2, sampled
  2. REMEMBER  mem.add_detections(dets); stats = mem.commit()         # fuse/dedup (atomic)
               changes = mem.changes(last_commit_ts)                  # positional arg
  3. (between ticks, never mid-ingest) mem.consolidate(); mem.decay() # hygiene
on goal / question:
  4. RETRIEVE  ctx = mem.serialize(format="prompt", max_tokens=B)  +  query()/semantic()
  5. REASON    out = llm(ctx, goal, tools=[C3 tools])                # plan / answer (cite ids)
  6. ACT       execute(out)  |  return answer
  7. WRITE-BACK if correction: mem.update(node_id, ...)              # close the loop
```

Perception (1–2) runs on a **sampled keyframe cadence** (Cosmos is seconds/clip,
not per-frame). Reasoning (4–6) runs on demand. Memory is the buffer between a
slow eye and an on-demand cortex — but on v0 both run in the *same thread* (§6).

**Clock note:** `changes()`/`stale()` key off observation timestamps. Stamp every
commit with one authoritative clock and pass `det.ts` consistently (if `det.ts`
is `None` it falls back to ingest wall-clock — don't mix clip-time and wall-time
in one batch). `moved()` measures first↔last displacement (net, not recent
motion) — note this when surfacing "what moved".

## 8. Why all three (ablation)

| Drop | What breaks |
|---|---|
| − Cosmos 3 | no perception / no 3D grounding from raw video |
| − SpatialMem | no persistence; forgets across clips/sessions (Cosmos's own gap); LLM re-reads raw frames every time, blows context |
| − LLM | no general planning / language / abstraction over the remembered world |

## 9. Store compatibility across sessions

Persistence across "clips, sessions, days" has hard, real constraints:

- **`embedding_dim` is locked at create time.** Reopening with a different
  encoder (CLIP-512 vs SigLIP-768) raises `SchemaMismatchError`. "Remember
  across sessions" requires the **same encoder forever**; the Cosmos adapter's
  CLIP-crop feature makes the encoder a load-bearing, immutable choice. To
  change it: re-embed into a new store.
- **Forward-only schema** (`SCHEMA_VERSION`, migrations dir): a store written by
  a newer lib will not open on an older one. Pin the lib for a long-lived store.
- **`aux` is not a column.** The adapter emits `aux={source, orient, box2d}`, but
  `nodes`/`observations` have no `aux` column → orientation/box2d are **dropped
  on persist** today. Answers cannot rely on 3D orientation across sessions
  until a schema bump adds a JSON `aux` column.

## 10. Security

| Surface | Risk | Mitigation |
|---|---|---|
| Scene text → LLM prompt | **Prompt injection** — object labels come from video the user doesn't control and flow verbatim into `to_prompt()` then the LLM; a label like `"ignore prior instructions…"` + C3 tool access is an exfil vector | Escape/delimit untrusted scene text in the prompt; whitelist which C3 tools an injected instruction can reach; treat tool args as untrusted |
| Data egress | `CosmosReasonVerbalizer` POSTs the serialized scene to `integrate.api.nvidia.com` with a Bearer key; Topology A/B both send scene data **off-box** | State it explicitly to users; offer a local path (Ollama / self-hosted NIM) for sensitive deployments |
| API keys | `NVIDIA_API_KEY` (and BYO-LLM keys) | env only; never persisted into `.smem`; redact in logs |
| Cosmos input video | untrusted media | treat as untrusted; validate frame shapes/dtypes |

## 11. Error handling & degradation

The architecture's strongest property — **memory degrades gracefully when
perception or reasoning is unavailable** — must be explicit and tested:

| Component down | Behavior |
|---|---|
| Cosmos NIM timeout / bad JSON | adapter raises, **skip the frame**, log, keep serving existing memory |
| Encoder missing | fail fast at **wiring time**, not per-frame (feature is required) |
| BYO-LLM unreachable / bad tool call | retry policy → fall back to a raw `serialize()` dump |
| `commit()` timeout (real `timeout_s=30`) | defined exception surface; caller retries or drops the batch |

## 12. Non-goals

Inherits SpatialMem's: **not** a SLAM (pose is upstream), **not** the action /
motion-planning layer (lives in the consumer/LLM), **not** a new VLM.
**v0 is single-camera, single-agent.** Multi-camera/agent needs primitives this
design does not yet have: per-observation camera/agent provenance, a single-
writer coordination story (the future gRPC/service façade owns the writer;
agents are clients), per-source pending buffers, and cross-agent extrinsic
calibration. Out of scope until those land.

## 13. Episodic representation (time + place)

The headline eval (§15) depends on episodic memory, so define it:

- An **episode = one session run** (store models `episodes(id, session,
  start_ts, end_ts)`; nodes carry `t_first/t_last`).
- Time-scoped + place-scoped retrieval: a `whats_in(region, at_ts=…|episode=…)`
  tool, and `serialize()` rendering a past snapshot or a **delta vs now**.
- This is the query path the "object seen in clip 1, asked in clip 9" eval
  exercises.

## 14. Observability

A wrong answer spans three black boxes, so thread one **tick/trace id** through
PERCEIVE→REMEMBER→REASON→ACT. Log each C3 tool call (name, args, returned
`node_id`s, latency). Record provenance: LLM answer → cited `node_id`s → source
observation/detection → Cosmos clip. Hook into existing `mem.stats()` /
`mem.dump()` / `spatialmem.events` logs.

## 15. Build phases

| Phase | Deliverable | Status |
|---|---|---|
| P0 | `CosmosReasonVerbalizer` (answer backend, Cosmos Reason 2 today) | ✅ shipped (core) |
| P1 | `Cosmos3PerceptionAdapter` (C1) + geometry lift + `ImageEncoder` — needs GPU + schema probe | ⬜ in `spatialmem-perception` |
| P2 | memory-as-tools (C3) — `SpatialMemTools` (schemas + dispatch, validated args) | ✅ shipped |
| P2+ | thin MCP server wrapping `SpatialMemTools` | ⬜ |
| P3 | active-perception loop (C4) — bounded | ⬜ |
| P4 | end-to-end agent demo: stream scene → multi-turn planning with memory | ⬜ |

## 16. Evaluation

- Reuse `bench.recall_at_k` for retrieval quality.
- **"Memory advantage" eval:** questions answerable *only* with cross-clip /
  cross-session persistence (object seen in clip 1, asked in clip 9) — Cosmos
  alone (bounded context) fails; Cosmos+SpatialMem passes. This number proves
  the architecture, not just the components.
- **Latency budget** per tick, reported separately: perception (sampled),
  reasoning (on demand), **and maintenance** (`consolidate()` is O(n²) and
  `candidates_near()` is a linear AABB scan until the planned R-tree lands —
  maintenance is the cost that scales worst across "days").

## 17. Cost model (rough, to make "B is default" defensible)

Assume 1 keyframe clip / 5 s and a 1500-token serialize budget:

- **Cosmos (perception):** $/clip via NIM, or GPU-hour amortized for self-host
  Nano 16B (workstation) / Super 64B (datacenter); gate on keyframes/events to
  cap it. Edge (~4B) is **not yet released**.
- **Reasoning, Topology B:** ~1500 input + few-hundred output tokens × $/Mtok of
  the chosen LLM, **per question** (not per frame) → typically << perception.
- **Topology A:** a System-2 VLM reason step per question on GPU → materially
  pricier than B's text LLM; this is the core reason B is the default.

(Plug real per-model rates when a target model is chosen.)

## 18. Worked example (Topology B, real shipped methods)

```python
import numpy as np
from spatialmem import SpatialMemory

def stub_llm(context: str, question: str) -> str:        # swap for Claude/GPT/Ollama
    return f"(reasoning over)\n{context}\nQ: {question}"

mem = SpatialMemory.open("home.smem", embedding_dim=512, encoder=my_clip)
# C1: Cosmos-perceived detections (here: pre-made) -> fuse
mem.add_detections([det_mug, det_table]); mem.commit()
# C2: serialize for the cortex
ctx = mem.serialize(format="prompt", max_tokens=1500)
# C3 (proposed wrapper) would expose query/semantic/spatial as tools; shipped today:
hits = mem.semantic("coffee mug")                        # -> [NodeHit(node_id=…, …)]
# REASON (Topology B): BYO LLM over ctx + tool results, cite node ids
print(stub_llm(ctx, "where is the mug?"))
# or one-shot via the shipped verbalizer path:
# mem = SpatialMemory.open("home.smem", embedding_dim=512, encoder=my_clip, verbalizer=cosmos_vb)
# print(mem.answer("where is the mug?"))
```

## 19. One-paragraph pitch

> Cosmos 3 sees and physically reasons about a video clip; SpatialMem remembers
> what it saw as a persistent, queryable 3D scene graph; an LLM plans and talks
> over that memory. Cosmos perceives, SpatialMem remembers, the LLM thinks — a
> brain whose hippocampus NVIDIA's own flagship report says is still missing.
