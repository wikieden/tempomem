> 🌐 **English** · [中文](../zh/01-POSITIONING.md)

# 01 · Positioning & Competitive Landscape

## The Empty Cell

| Capability | ConceptGraphs | Hydra | Open3DSG | DovSG | nvblox | eMEM | Mem0 | **Chronotope** |
|---|---|---|---|---|---|---|---|---|
| Open-vocab semantics | ✅ | △ | ✅ | ✅ | △ | ✗ | ✗ | **✅** |
| Hierarchical scene graph | ✗ | ✅ | ✗ | ✗ | ✗ | △ | ✅ (text) | **✅** |
| Incremental / streaming | ✗ | ✅ | ✗ | △ | ✅ | ✅ | ✅ | **✅** |
| Persistent (file / DB) | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | ✅ | **✅** |
| NL query API | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | ✅ | **✅** |
| `pip install` clean | ✗ | ✗ | ✗ | ✗ | △ | ✅ | ✅ | **✅** |
| Framework-agnostic (no ROS lock) | ✅ | ✗ | ✅ | ✗ | △ | ✗ | ✅ | **✅** |
| LLM-grade serialization | ✗ | ✗ | ✗ | ✗ | ✗ | △ | ✅ | **✅** |

No row in the existing market hits all eight cells. That's the wedge.

## Closest Competitors

### eMEM (Automatika Robotics) — direct
- **Strength:** First mover, real spatial memory, runs on Jetson, integrated with their EMOS stack.
- **Weakness:** ROS 2-locked (Python core but every example assumes ROS topics); no open-vocab semantics (closed label set); 2★ / niche brand; license fine but no community.
- **Our edge:** Framework-agnostic core (ROS 2 is an adapter, not a dependency). CLIP/SigLIP-grounded open-vocab from day one. Faster `add_frame` → `query` path with no ROS install. Better DX.

### Mem0 — adjacent, NOT competitor
- **Posture:** ally / inspiration. They own text memory. We complement.
- **Strategy:** ship a `mem0` adapter so a Mem0-using agent can plug Chronotope in as the "spatial backend" and unified memory works across modalities. Co-marketing potential.

### ConceptGraphs / Hydra / Open3DSG / DovSG — upstream
- **Posture:** these are *perception backends* we can call from inside our ingest layer. Not competitors — suppliers.
- **Strategy:** pick **ConceptGraphs** as the v0 fusion backend (Apache-2.0, RGB-D, well-cited, demo-friendly). Add Hydra and nvblox-feature adapters later.

### NVIDIA Isaac / GR00T / Cosmos — adjacent platform
- **Posture:** they ship the perception + VLA brain. They explicitly do not ship persistent spatial memory.
- **Strategy:** publish an Isaac ROS / GR00T integration recipe. Become the "memory layer GR00T forgot."
- **Validation (2026-06):** NVIDIA's own **Cosmos 3** technical report (138 pp, GTC Taipei) names the gap as the field's open problem — *"physical intelligence needs more than recognition … it requires temporally persistent state, spatial grounding tied to objects and agents … a maintained, actionable scene estimate."* Cosmos 3 itself ships none of it: a bounded 74K-token generator context, ephemeral per-request KV-cache, zero mentions of "spatial memory" or "cross-session." Their flagship Physical-AI model **defines our thesis as unsolved** — the strongest possible third-party endorsement of the wedge.
- **Seam, not threat:** Cosmos 3's Reasoner emits *structured camera-frame 3D boxes (unified JSON) + estimated metric ego-pose* — i.e. it is a strong **upstream detection source** for our ingest contract (a `Cosmos3PerceptionAdapter`), not a memory competitor. We sit on top, persisting what it perceives.

## Three-sentence pitch (memorize)

> Mem0 cracked text memory for agents. NVIDIA cracked perception for embodied AI. Nothing cracked the spatial memory in between — every robotics lab rebuilds it. Chronotope is the open, pip-installable, framework-agnostic spatial memory layer that turns RGB-D streams into a persistent, LLM-queryable scene graph.

## Five-Year Threat Model

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mem0 ships a 3D extension | medium | Move fast, build the brand, court an acquisition offer or stay friendly partner |
| NVIDIA absorbs into Isaac | medium | Stay framework-agnostic; their stack is CUDA-locked, ours runs on Mac/CPU/Jetson |
| eMEM gets traction | low | Their ROS lock-in is a permanent ceiling for non-robotics users |
| New academic SOTA scene graph | high | We're a *layer*, not a backend — swap them in via adapter |
| LLMs gain world models that obviate explicit memory | medium-term | Plays to our favor: the memory tokens still have to come from somewhere persistent |
