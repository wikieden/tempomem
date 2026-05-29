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
             readonly: bool = False) -> "SpatialMemory": ...

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

def add_frame(self, rgb: np.ndarray, depth: np.ndarray,
              pose: np.ndarray, intrinsics: np.ndarray,
              ts: float | None = None,
              *, episode: str | None = None) -> int:
    """Run configured ingest adapter. Returns frame id. Non-blocking."""

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
          intent: Literal["auto", "semantic", "spatial", "temporal"] = "auto",
          k_hops: int = 0) -> QueryResult: ...

def semantic(self, text: str, *, k: int = 10) -> list[NodeHit]: ...
def spatial(self, *, near: tuple[float,float,float] | None = None,
            radius: float | None = None,
            inside_bbox: tuple[tuple[float,float,float], tuple[float,float,float]] | None = None,
            k: int = 100) -> list[NodeHit]: ...
def recent(self, *, n: int = 10, episode: str | None = None) -> list[NodeHit]: ...

def serialize(self, *, format: Literal["prompt", "json", "dot"] = "prompt",
              root: int | None = None,
              k_hops: int = 2,
              max_tokens: int | None = None) -> str: ...
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
    subgraph: SceneSubgraph | None      # populated when k_hops > 0
    debug: Mapping[str, JsonValue]
```

## Lifecycle / Maintenance

```python
def forget(self, node_id: int) -> None: ...
def decay(self, *, half_life_s: float, floor: float = 0.0) -> int: ...
def stats(self) -> StoreStats: ...
def dump(self, path: str | os.PathLike, *, format: Literal["json"] = "json") -> None: ...
```

## Configuration

`SpatialMemory.open(..., config=SpatialMemConfig(...))`. All thresholds live on the config object, never as method kwargs:

```python
@dataclass
class SpatialMemConfig:
    fusion: FusionConfig = field(default_factory=FusionConfig)
    query:  QueryConfig  = field(default_factory=QueryConfig)
    persist: PersistConfig = field(default_factory=PersistConfig)
```

Loaded from `~/.config/spatialmem/config.toml` if present; overridable via env (`SPATIALMEM_*`).

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
