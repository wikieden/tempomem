# spec · Scene-Graph Schema & Storage

Normative. Bumping any field in this doc requires a migration script in `spatialmem/persist/migrations/`.

## Logical Model

```
Episode ─┐
         ├── Observation ──┐
         │                 │ (many-to-many)
Node ────┴── node_obs ─────┘
  │
  ├── Edge (typed) ── Node
  └── Feature vector
```

## Tables

### `meta`
| col | type | note |
|---|---|---|
| key | TEXT PK | |
| value | TEXT | |

Required keys: `schema_version` (int as text), `embedding_dim`, `created_at`, `creator_version`.

### `episodes`
| col | type | note |
|---|---|---|
| id | INTEGER PK | |
| session | TEXT | user-supplied or `default` |
| label | TEXT | nullable |
| start_ts | REAL | UNIX seconds |
| end_ts | REAL | nullable until closed |

### `observations`
| col | type | note |
|---|---|---|
| id | INTEGER PK | |
| episode_id | INTEGER FK | |
| ts | REAL | |
| label | TEXT | open-vocab |
| confidence | REAL | 0..1 |
| center_x, center_y, center_z | REAL | meters |
| bbox_min_x, bbox_min_y, bbox_min_z | REAL | |
| bbox_max_x, bbox_max_y, bbox_max_z | REAL | |
| mask_rle | BLOB | nullable |
| aux | TEXT | JSON |

Index: `(episode_id, ts)`.

### `obs_features` *(virtual, sqlite-vec)*
`obs_id → vec(D)`, D = `meta.embedding_dim`.

### `nodes`
| col | type | note |
|---|---|---|
| id | INTEGER PK | |
| type | TEXT | `object` \| `place` \| `room` \| `floor` |
| label | TEXT | canonical label (most-frequent among observations) |
| labels_json | TEXT | full label distribution `[[label, weight], ...]` |
| confidence | REAL | aggregated |
| centroid_x, centroid_y, centroid_z | REAL | running mean (conf-weighted) |
| bbox_min_*, bbox_max_* | REAL | aggregated |
| n_obs | INTEGER | observation count |
| t_first | REAL | |
| t_last | REAL | |
| parent_id | INTEGER FK NULL | hierarchical containment |

Index: `(type)`, `(label)`, `(t_last DESC)`.

### `node_features` *(virtual, sqlite-vec)*
`node_id → vec(D)` (centroid of constituent observation features, L2-normalized).

### `node_geom` *(virtual, rtree)*
`node_id → (xmin, xmax, ymin, ymax, zmin, zmax)`.

### `edges`
| col | type | note |
|---|---|---|
| id | INTEGER PK | |
| src | INTEGER FK | |
| dst | INTEGER FK | |
| type | TEXT | enum below |
| confidence | REAL | |
| t_last | REAL | |
| aux | TEXT | JSON |

Unique: `(src, dst, type)`.

### `node_obs`
| col | type | note |
|---|---|---|
| node_id | INTEGER FK | |
| obs_id | INTEGER FK | |
| ts | REAL | |

PK: `(node_id, obs_id)`.

## Enums

### Node `type`
`object`, `place`, `room`, `floor`.

### Edge `type`
`on`, `inside`, `under`, `near`, `part_of`, `same_room_as`, `co_observed`, `temporal_before`.

Adding a new edge type bumps `schema_version`.

## JSON Export Schema (`mem.dump`)

```jsonc
{
  "schema_version": 1,
  "embedding_dim": 512,
  "episodes": [{ "id": 1, "session": "default", "start_ts": 1735.0, "end_ts": 1900.0 }],
  "nodes": [
    {
      "id": 42, "type": "object", "label": "mug",
      "labels": [["mug", 0.81], ["cup", 0.19]],
      "confidence": 0.87,
      "centroid": [1.21, 0.32, 0.94],
      "bbox": [[1.1,0.2,0.8],[1.3,0.4,1.0]],
      "n_obs": 7, "t_first": 1740.1, "t_last": 1899.4,
      "parent_id": 12
    }
  ],
  "edges": [{ "src": 42, "dst": 12, "type": "on", "confidence": 0.92, "t_last": 1899.4 }]
}
```

## Prompt Serialization (`format="prompt"`)

Indented, token-efficient, deterministic. Default template:

```
SCENE (root=kitchen, k_hops=1, ts=1900.0)
  room#12 "kitchen"
    on:
      object#42 "mug"            @[1.21, 0.32, 0.94]   t_last=1899.4  conf=0.87
      object#43 "kettle"         @[1.55, 0.40, 0.93]   t_last=1898.1  conf=0.91
    near:
      object#44 "fridge"         @[2.40, 0.00, 1.20]   t_last=1885.0  conf=0.95
```

Rules:
- Header line with root + k_hops + current ts.
- Two-space indent per hop.
- One node per line: `type#id "label"  @[x,y,z]  t_last=...  conf=...`.
- Edges grouped under `<type>:` headers, alphabetical.
- Truncation: if `max_tokens` set, breadth-first prune by `conf * recency_score`; mark with `... (N omitted)`.

## Versioning

- `schema_version` starts at `1` at first public release.
- Migrations are forward-only Python scripts in `spatialmem/persist/migrations/NNN_*.py`. Each defines `up(conn)`. No `down()`.
- `open()` runs all pending migrations under a single transaction. Failure rolls back; file is unchanged.
