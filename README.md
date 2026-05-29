# SpatialMem

Spatial memory layer for AI agents. Turn RGB-D / point clouds into a persistent, queryable, LLM-native 3D scene graph — in one pip install.

> **Positioning:** Mem0 for 3D space. ConceptGraphs / Hydra do perception; Mem0 does text memory; nothing today bridges perception → persistent, agent-queryable spatial memory. We fill that slot.

## Quickstart

```python
from spatialmem import SpatialMemory, Detection

# Detections-in (BYO perception): supply label + 3D bbox + feature vector.
mem = SpatialMemory.open("kitchen.smem", embedding_dim=512)
mem.add_detections([
    Detection("mug", feat, center_xyz=(1.2, 0.3, 0.9),
              bbox_min=(1.15, 0.25, 0.85), bbox_max=(1.25, 0.35, 0.95)),
])
mem.commit()                       # runs the fusion arbiter — incremental dedup

mem.recent(n=5)                                    # temporal
mem.spatial(near=(1.0, 0.0, 1.0), radius=2.0)      # spatial
prompt = mem.serialize(format="prompt")            # graph -> compact LLM text

# Natural-language semantic search + answer need an encoder/verbalizer:
mem = SpatialMemory.open("kitchen.smem", embedding_dim=512,
                         encoder=my_clip, verbalizer=my_llm)
mem.semantic("coffee mug")                         # cosine over node features
mem.answer("where is the mug?")                    # retrieve -> prompt -> BYO LLM

# Streaming RGB-D (needs a PerceptionAdapter; ConceptGraphs adapter is WIP):
# mem = SpatialMemory.open(..., adapter=MyAdapter())
# mem.add_frame(rgb, depth, pose); mem.commit()
```

Runnable, no GPU: `python examples/01_quickstart.py` and `examples/02_query_and_answer.py`.

## What works today

| Capability | Status |
|---|---|
| Single-file `.smem` SQLite store, persist/reopen | ✅ |
| `add_detections` + incremental fusion (dedup, merge, reject) | ✅ |
| Spatial / temporal / keyword query | ✅ |
| Semantic query via BYO `Encoder` (`OpenClipEncoder` in `[clip]`) | ✅ |
| sqlite-vec ANN index (`[vec]`), linear fallback | ✅ |
| `answer()` via BYO `Verbalizer` (OpenAI / Anthropic / Ollama) | ✅ |
| `decay()` + `forget()` + `resplit()` memory hygiene | ✅ |
| `serialize(format="prompt"/"json")` for LLM hand-off | ✅ |
| `recall_at_k` eval harness | ✅ |
| Read-only HTML viewer — `spatialmem viz store.smem -o scene.html` | ✅ |
| RGB-D `add_frame` via `PerceptionAdapter` protocol | ✅ seam; ConceptGraphs adapter WIP (CUDA) |

Core install is **numpy-only**. Heavy backends live behind extras: `[clip]`, `[vec]`, `[perception]`.

## Repo Layout

```
docs/        product + engineering decisions (read first)
spec/        normative API / schema specs
src/spatialmem/   Python package (lib)
examples/    runnable demos (real + simulated)
tests/
```

## Status

Pre-alpha. Public design phase. Read in order:

- [docs/00-VISION.md](docs/00-VISION.md) — what & why
- [docs/01-POSITIONING.md](docs/01-POSITIONING.md) — competitors & wedge
- [docs/02-ARCHITECTURE.md](docs/02-ARCHITECTURE.md) — layers & dataflow
- [docs/03-ROADMAP.md](docs/03-ROADMAP.md) — milestones
- [docs/04-MVP-SCOPE.md](docs/04-MVP-SCOPE.md) — first deliverable
- [spec/API.md](spec/API.md) — Python SDK contract
- [spec/SCHEMA.md](spec/SCHEMA.md) — scene-graph schema
- [spec/FUSION-ARBITER.md](spec/FUSION-ARBITER.md) — node merging algorithm
- [spec/QUERY-ROUTER.md](spec/QUERY-ROUTER.md) — query → retrieval
- [spec/ENGINEERING.md](spec/ENGINEERING.md) — coding standards / CI

## License

Apache-2.0 — see [LICENSE](LICENSE).
