# SpatialMem

Spatial memory layer for AI agents. Turn RGB-D / point clouds into a persistent, queryable, LLM-native 3D scene graph — in one pip install.

> **Positioning:** Mem0 for 3D space. ConceptGraphs / Hydra do perception; Mem0 does text memory; nothing today bridges perception → persistent, agent-queryable spatial memory. We fill that slot.

## Quickstart (target API)

```python
from spatialmem import SpatialMemory

mem = SpatialMemory.open("./demo.smem")          # SQLite + vector backend

mem.add_frame(rgb, depth, pose, ts=...)          # streaming ingest
mem.commit()                                     # flush graph updates

# Natural-language spatial query → text answer + grounded node ids
ans = mem.query("Where was the coffee mug last seen?")
# → "On the kitchen counter, ~2.3m from the fridge, 14:02."
#   ans.nodes = [Node(id=42, label="mug", center=[1.2, 0.3, 0.9], ...)]

# Or hand the graph to any LLM agent
graph_text = mem.serialize(format="prompt", k_hops=2, root="kitchen")
```

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

Apache-2.0 (planned).
