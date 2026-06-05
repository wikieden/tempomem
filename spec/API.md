# spec · API (Python SDK)

Normative. Anything not listed is non-public and may change without notice.

## Conventions

- Python ≥ 3.10. Type hints required on all public symbols.
- Public symbols re-exported from `spatialmem` top-level.
- All public methods raise `SpatialMemError` (or subclass) on failure. No bare exceptions.
- Coordinates: right-handed, **meters**, world frame. Pose = 4×4 numpy float32 in `Tcw` (camera-from-world) form.
- Timestamps: float seconds (UNIX epoch) **or** monotonic — caller-consistent within a store.
- Embeddings: float32, L2-normalized, fixed dim per store (set at `open`).

## Top-Level

```python
class SpatialMemory:
    @classmethod
    def open(cls, path: str | os.PathLike, *,
             embedding_dim: int = 512,
             create: bool = True,
             readonly: bool = False,
             config: SpatialMemConfig | None = None,
             encoder: Encoder | None = None,           # semantic query / answer
             verbalizer: Verbalizer | None = None,     # answer()
             adapter: PerceptionAdapter | None = None  # add_frame()
             ) -> "SpatialMemory": ...

    def close(self) -> None: ...
    def __enter__(self) -> "SpatialMemory": ...
    def __exit__(self, *a) -> None: ...
```

`open` is idempotent. If file exists, `embedding_dim` must match stored value or raise `SchemaMismatchError`.

## Ingest

```python
def add_detections(self, dets: Sequence[Detection],
                   *, episode: str | None = None) -> List[int]:
    """Insert observations. Returns assigned observation ids. Non-blocking."""

def add_frame(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray,
              *, intrinsics: np.ndarray | None = None,
              adapter: PerceptionAdapter | None = None,
              episode: str | None = None) -> list[int]:
    """Route a posed RGB-D frame through a PerceptionAdapter to detections,
    then add_detections. Adapter from `adapter=` or open(adapter=).
    Returns assigned observation ids. Non-blocking (still needs commit())."""

def commit(self, *, timeout_s: float = 30.0) -> CommitStats:
    """Flush queued observations through the arbiter and fsync."""
```

`Detection` is a frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class Detection:
    label: str                          # open-vocab string
    feature: np.ndarray                 # shape (D,), L2-normalized float32
    center_xyz: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    confidence: float = 1.0             # 0..1
    mask_rle: bytes | None = None       # optional COCO RLE
    ts: float | None = None             # default: now
    aux: Mapping[str, JsonValue] = field(default_factory=dict)
```

## Query

```python
def query(self, text: str, *, k: int = 10,
          intent: Literal["auto", "semantic", "spatial", "temporal"] = "auto"
          ) -> QueryResult: ...
    # "auto" routes by keyword heuristic; semantic/hybrid use the encoder when
    # configured, else label-keyword fallback. (k_hops subgraph: planned.)

def semantic(self, text: str, *, k: int = 10) -> list[NodeHit]: ...
    # encoder-backed cosine when open(encoder=...); else label keyword match.

def answer(self, query: str, *, k: int = 8,
           verbalizer: Verbalizer | None = None) -> str: ...
    # retrieve -> serialize(prompt) -> BYO Verbalizer. Raises QueryError if none.

def spatial(self, *, near: tuple[float,float,float] | None = None,
            radius: float | None = None,
            k: int = 100) -> list[NodeHit]: ...                 # inside_bbox: planned
def recent(self, *, n: int = 10) -> list[NodeHit]: ...           # episode filter: planned

def serialize(self, *, format: Literal["prompt", "json"] = "prompt",
              root: int | None = None,
              k_hops: int = 2,
              relations: bool = True,
              max_tokens: int | None = None) -> str: ...          # "dot": planned
    # prompt: hierarchy-indented; each node line gets a `| <rel> <label>#<id>`
    # suffix from its edges when relations=True (after relate()). max_tokens caps
    # output most-recent-first with an explicit `… (N more omitted)` marker.
```

Return shapes:

```python
@dataclass
class NodeHit:
    id: int
    label: str
    center_xyz: tuple[float, float, float]
    confidence: float
    score: float                        # retriever score (higher = better)
    t_first: float
    t_last: float

@dataclass
class QueryResult:
    nodes: list[NodeHit]
    intent_used: Literal["semantic", "spatial", "temporal", "hybrid"]
    debug: Mapping[str, JsonValue]      # e.g. {"text": ..., "encoder": True}
    # subgraph (k_hops expansion): planned
```

## Lifecycle / Maintenance

```python
def forget(self, node_id: int) -> None: ...

def decay(self, *, half_life_days: float = 30.0, min_conf: float = 0.1,
          now: float | None = None) -> tuple[int, int]: ...
    # conf' = conf * 0.5 ** (age_days / half_life_days); prune below min_conf.
    # Returns (n_decayed, n_pruned).

def resplit(self) -> tuple[int, int]: ...
    # Split nodes whose member observations form two separated clusters
    # (config tau_split_m / min_split_obs). Returns (nodes_split, new_nodes).

def stats(self) -> StoreStats: ...
```

JSON export is via `serialize(format="json")`. A standalone `dump()` is planned.

## Hierarchy

```python
def define_region(self, label: str,
                  bbox_min: tuple[float, float, float],
                  bbox_max: tuple[float, float, float],
                  *, type_: str = "room") -> int: ...
    # Create (or, by (label, type_), redefine) a region node over an AABB and set
    # parent_id on every object node whose centroid is inside. Region feature =
    # encoder.encode_text(label) when open(encoder=...), else mean of child
    # features. Single-parent membership: on overlap, last definition wins.
    # Returns the region node id.

def contents(self, region: int | str) -> list[NodeHit]: ...
    # Child nodes of a region, addressed by node id or by region label.
```

Regions are structure, not observations: `decay()` never ages or prunes them,
and `forget()` reparents a deleted node's children to its parent (no dangling
`parent_id`).

## Relations

```python
def relate(self, *, near_m: float = 0.6, on_gap_m: float = 0.08) -> int: ...
    # Infer geometric relations over object nodes and store them as `edges`:
    #   near  — centroids within near_m (symmetric)
    #   on    — A's base near B's top with x/y overlap, A above B
    #   under — inverse of every `on`
    # Idempotent (clears prior auto edges first). Returns edges written.

def related(self, node: int | str, *, rel: str | None = None
            ) -> list[tuple[NodeHit, str]]: ...
    # Neighbors of a node (by id or object label) as (neighbor, relation_type),
    # optionally filtered to one relation type.
```

## Update / History

```python
def update(self, node_id: int, *,
           label: str | None = None,
           center_xyz: tuple[float, float, float] | None = None,
           confidence: float | None = None) -> None: ...
    # Correct a node in place. Only given fields change; moving center_xyz
    # shifts the bbox by the same delta (extent preserved). Raises StoreError
    # for a missing node or confidence outside [0, 1].

def history(self, node_id: int) -> list[Observation]: ...
    # Time-ordered observation trail behind a node — every fused sighting with
    # its ts + position. "Last seen" is history(node_id)[-1].

def moved(self, node_id: int) -> float: ...
    # Displacement (m) of a node between its first and last observation. 0 for <2 obs.
def changes(self, since_ts: float) -> ChangeSet: ...
    # ChangeSet(new, seen_again): nodes that first appeared / were re-observed since ts.
def stale(self, before_ts: float) -> list[NodeHit]: ...
    # Nodes not observed since before_ts — candidates for "gone".

def merge(self, other: str | os.PathLike, *, episode: str | None = None) -> CommitStats: ...
    # Fold another .smem store's object nodes into this one through fusion:
    # shared objects dedupe to one node, new objects are added. Re-entering a
    # space continues the memory. Embedding dims must match; regions not merged.
```

## Configuration

`SpatialMemory.open(..., config=SpatialMemConfig(...))`. All thresholds live on the config object, never as method kwargs:

```python
@dataclass(frozen=True, slots=True)
class SpatialMemConfig:
    fusion: FusionConfig = field(default_factory=FusionConfig)
    # query / persist sub-configs: planned
```

`FusionConfig` carries all fusion thresholds (weights, `tau_merge`, `tau_obs`,
`tau_split_m`, `min_split_obs`, `centroid_alpha`, …). TOML/env loading is planned.

## Pluggable backends (protocols)

Duck-typed; supply any object matching the shape. Core stays numpy-only.

```python
class Encoder(Protocol):                 # spatialmem.encoders
    @property
    def dim(self) -> int: ...
    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...  # (N, dim) L2-norm

class Verbalizer(Protocol):              # spatialmem.verbalize
    def complete(self, prompt: str) -> str: ...

class PerceptionAdapter(Protocol):       # spatialmem.perception
    def process_frame(self, rgb, depth, pose,
                      intrinsics=None) -> list[Detection]: ...
```

Reference impls: `OpenClipEncoder` (`[clip]` extra). ConceptGraphs perception
adapter is WIP (`[perception]`, CUDA).

## Errors

```
SpatialMemError
├── SchemaMismatchError
├── IngestError
│   ├── BadDetectionError
│   └── AdapterError
├── QueryError
└── StoreError
```

## Stability Promise (post v0.1.0)

- `open`, `add_detections`, `add_frame`, `commit`, `query`, `serialize`, `close`, `Detection`, `NodeHit`, `QueryResult` → **stable**. SemVer-protected.
- Everything else → `experimental`. Subject to change in any minor release until stabilized.

## What this API does NOT do

- No motion planning / control
- No multi-tenant auth (single user/process)
- No streaming subscriptions (poll `stats()` or upgrade to gRPC façade in M4)
- No clustering — single `.smem` file is the unit
