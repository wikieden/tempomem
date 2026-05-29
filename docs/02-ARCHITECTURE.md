# 02 · Architecture

## Layered Stack

```
┌──────────────────────────────────────────────────────────────┐
│  L5  Agent / LLM           ← consumer (LangChain, Mem0, raw) │
├──────────────────────────────────────────────────────────────┤
│  L4  Query Router          NL → (semantic | spatial | temp)  │
│       └─ serializer        graph → prompt text               │
├──────────────────────────────────────────────────────────────┤
│  L3  Memory Store          objects · places · rooms · events │
│       ├─ Fusion Arbiter    dedup / merge / split             │
│       └─ Decay / Forget    confidence + TTL                  │
├──────────────────────────────────────────────────────────────┤
│  L2  Persistence           SQLite + sqlite-vec + R-tree      │
├──────────────────────────────────────────────────────────────┤
│  L1  Ingest Adapters       conceptgraphs · hydra · custom    │
├──────────────────────────────────────────────────────────────┤
│  L0  Perception (external) RGB-D · pose · CLIP/SigLIP feats  │
└──────────────────────────────────────────────────────────────┘
```

Each layer is a single Python module with a documented Protocol. Replacing one does not cascade.

## Core Concepts

- **Frame** — atomic ingest unit: `(rgb, depth, intrinsics, pose, ts)` *or* `(detections, ts)`.
- **Observation** — what a single frame says about a single object: 2D mask / 3D points + features + label distribution + bbox + ts.
- **Node** — a stable scene-graph entity. Types: `Object`, `Place`, `Room`, `Floor`. Holds rolling-aggregated geometry + feature centroid + history of observation ids.
- **Edge** — typed relation: `on`, `inside`, `near`, `part_of`, `same_room_as`, `temporal_before`. Edges carry confidence ∈ [0,1] and last_seen ts.
- **Episode** — a contiguous run of frames sharing a session id. Episodes commit and can be replayed.

## Data Flow (happy path)

```
add_frame(rgb,d,pose) ──▶ Ingest Adapter (ConceptGraphs)
                              │
                              ▼
                       observations[] (per-frame)
                              │
                              ▼
                       Fusion Arbiter
                       ├─ candidate match (KNN on centroid + IoU3D)
                       ├─ score (geom + CLIP cos + label agree)
                       ├─ decision: merge | new node | reject
                       ▼
                       Memory Store mutation tx
                              │
                              ▼
                       Persistence write (single fsync per commit)
```

Query path:

```
query("where is the red mug?") ──▶ Query Router
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        spatial intent?         semantic intent?      temporal intent?
        (near/in/under)         ("red mug")           ("last seen")
                │                      │                      │
                ▼                      ▼                      ▼
        R-tree range scan       sqlite-vec ANN          ts index
                └──────────────────────┼──────────────────────┘
                                       ▼
                              candidate node set
                                       │
                                       ▼
                          k-hop subgraph extract
                                       │
                                       ▼
                       LLM verbalizer  ──▶  Answer + cited node ids
```

## Module Map (planned)

| Module | Responsibility | LOC budget v0 |
|---|---|---|
| `spatialmem.frame` | Frame / Observation dataclasses | <200 |
| `spatialmem.adapters.conceptgraphs` | RGB-D → observations via ConceptGraphs backend | <600 |
| `spatialmem.adapters.detections` | Pre-detected observations (BYO perception) | <200 |
| `spatialmem.fusion` | Arbiter + match scoring | <500 |
| `spatialmem.store` | Node / Edge / Episode CRUD over SQLite | <600 |
| `spatialmem.persist` | Schema migrations, sqlite-vec, R-tree | <400 |
| `spatialmem.query` | Router + retrievers + verbalizer | <600 |
| `spatialmem.serialize` | Graph → prompt text / JSON / DOT | <300 |
| `spatialmem.llm` | BYO-LLM Protocol + thin wrappers | <200 |
| `spatialmem.viz` | Web viewer (separate optional dep) | <800 |
| `spatialmem.bridges.ros2` | optional ROS 2 node | <400 |
| `spatialmem.bridges.mem0` | optional Mem0 spatial-backend shim | <200 |

Total core ~3.5k LOC. Aggressively small on purpose.

## Persistence Layout (single `.smem` file)

```
SQLite file
├── meta(table)                schema version, created_at, embedding_dim
├── episodes(table)            id, session, start_ts, end_ts, label
├── observations(table)        id, episode, ts, pose, bbox, label, conf
├── obs_features (vec)         sqlite-vec virtual table (obs_id → vec)
├── nodes(table)               id, type, label, centroid, bbox, conf, t_first, t_last
├── node_features (vec)        sqlite-vec (node_id → centroid embedding)
├── node_geom (rtree)          node_id → (xmin,xmax,ymin,ymax,zmin,zmax)
├── edges(table)               id, src, dst, type, conf, t_last
└── node_obs(table)            node_id ↔ observation_id (many:many)
```

One file = portable, diffable demo state, easy backup. No external services in v0.

## Threading & Concurrency

- v0: single-writer, async ingest queue. `add_frame` returns immediately; arbiter runs in a worker; `commit()` joins.
- Query path holds a read snapshot (SQLite WAL).
- v1: multi-process via gRPC server façade.

## Hard Dependencies (v0)

`numpy`, `scipy.spatial`, `sqlite-vec`, `pillow`, `pydantic`. CLIP/SigLIP optional via `[clip]` extra. ConceptGraphs adapter installed via `[conceptgraphs]` extra and pinned to a tested commit.

Everything else (Torch, ROS, CUDA) is **optional extras**. `pip install spatialmem` on a Mac with no CUDA must work end-to-end with the `detections` adapter.

## Observability

- Structured JSON logs (`spatialmem.events`) with one event per fusion decision.
- `mem.stats()` returns `{n_nodes, n_edges, n_obs, last_commit_ms, store_bytes}`.
- `mem.dump(path)` exports full graph as JSON for debugging.
