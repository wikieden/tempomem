# Design — `Cosmos3PerceptionAdapter` (path B)

**Status:** draft / GPU-blocked. **Track:** DEV-PLAN Phase C (learned perception).
**Depends on:** a CUDA box (Cosmos 3 Nano 16B / Super 64B) or a Cosmos NIM endpoint.

**Built so far (CPU, schema-independent):**
- ✅ `spatialmem.geometry` — `transform_points` / `oriented_box_corners` /
  `world_aabb_from_obb` (camera-frame OBB → world AABB), pure numpy, unit-tested.
- ✅ `spatialmem.ImageEncoder` protocol + `OpenClipEncoder.encode_image` — the
  per-object feature path (Cosmos emits no embedding).

**Still GPU-/schema-gated:** the Cosmos call + JSON `_parse_boxes` + the
adapter wiring — held until the box schema is probed on real weights.

## Goal

Let `add_frame(rgb, depth, pose)` use **NVIDIA Cosmos 3** as the perception
backend. The adapter turns one posed RGB frame into `list[Detection]` in world
coordinates, so the existing fusion → memory → query pipeline runs unchanged.

This is the *perception seam*, not the memory layer. SpatialMem still owns
persistence, fusion, and query. Cosmos perceives; we remember.

## Why this is now realistic (2026-06 finding)

Earlier we assumed a VLM gives only "2D boxes in text," forcing a full
depth + SLAM + CLIP + back-projection stack. The Cosmos 3 technical report
refutes that for Cosmos 3 specifically:

- Reasoner emits **structured camera-frame 3D boxes** — each object as
  `{label, center, dimensions, orientation (roll/pitch/yaw)}` in a **unified
  JSON** format (Sec 3.1.2), not free text.
- `Cosmos3-Nano` can **estimate metric-scale ego-pose** (inverse dynamics).

So Cosmos 3 supplies ~70% of a `Detection` (label + metric 3D box, camera
frame). The adapter only has to: (1) lift camera-frame → world-frame, (2) add a
per-object feature vector (Cosmos emits none), (3) parse the JSON robustly.

## Contract to satisfy

```python
# src/spatialmem/perception.py
class PerceptionAdapter(Protocol):
    def process_frame(self, rgb, depth, pose, intrinsics=None) -> list[Detection]: ...

# src/spatialmem/frame.py — the target value object
Detection(label, feature, center_xyz, bbox_min, bbox_max,
          confidence=1.0, mask_rle=None, ts=None, aux={})
#   feature: L2-normalized np.ndarray (dim == store embedding_dim)
#   center_xyz / bbox_*: world-frame meters, right-handed
```

## Pipeline (`process_frame`)

```
rgb ──► Cosmos 3 Reasoner (NIM or local Cosmos3OmniPipeline)
          prompt: "detect objects; per object return JSON
                   {label, center[x,y,z] m (camera), size[l,w,h] m,
                    orient[roll,pitch,yaw] rad, box2d[x1,y1,x2,y2]}"
          ──► parse JSON  (robust extract, like cosmos.strip_reasoning)
                │
   pose (arg) ──┤  cam→world 4×4   (prefer add_frame's pose;
   or Cosmos    │                   fall back to Cosmos ego-pose estimate)
   ego-pose ────┘
                ▼
   per object:
     center_world = pose @ center_cam
     bbox_min/max = world AABB of the oriented box's 8 corners
     feature      = encoder.encode_image(crop(rgb, box2d))   # CLIP/SigLIP tower
     Detection(label, feature, center_world, bbox_min, bbox_max,
               confidence, ts, aux={"source":"cosmos3",
                                    "orient": (r,p,y), "box2d": (...)})
```

Pure-numpy geometry (transform + oriented-box → AABB) is **CPU-testable** with a
fake reasoner; only the Cosmos call needs a GPU.

## Interface sketch

```python
class Cosmos3PerceptionAdapter:  # implements PerceptionAdapter
    def __init__(
        self,
        *,
        reasoner,                 # Cosmos 3 client: rgb -> raw JSON string
        encoder,                  # image-capable Encoder (see protocol gap)
        pose_source="arg",        # "arg" (use add_frame pose) | "cosmos" (estimate)
        confidence=0.8,           # prior; Cosmos per-box score undocumented
        grounding_prompt=_DEFAULT,
        transport=None,           # injection seam for offline tests
    ): ...

    def process_frame(self, rgb, depth, pose, intrinsics=None) -> list[Detection]:
        boxes = self._detect(rgb)                       # Cosmos -> [_Box3D]
        c2w = pose if self.pose_source == "arg" else self._estimate_pose(rgb)
        return [self._to_detection(b, c2w, rgb) for b in boxes]
```

Packaging: behind a `[cosmos]` (or reuse `[perception]`) extra; never a core
dependency. Default to a hosted NIM so no local weights are required.

## Required upstream change — `Encoder.encode_image`

The `Encoder` protocol today is text-only:

```python
class Encoder(Protocol):
    @property
    def dim(self) -> int: ...
    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...
```

Cosmos emits no per-object embedding, so the adapter must encode the **image
crop**. Add an optional image method (CLIP/SigLIP already have an image tower):

```python
    def encode_image(self, images: Sequence[np.ndarray]) -> np.ndarray: ...  # (N, dim) L2-norm
```

`OpenClipEncoder` (`[clip]`) gains `encode_image`; the adapter requires an
encoder that implements it (raise a clear error otherwise). This is the one
non-trivial core touch path B needs.

## Coordinate conventions (must confirm on real weights)

- SpatialMem world: right-handed, meters. `pose` = camera→world 4×4.
- Cosmos camera-frame axis convention is **undocumented** — verify empirically
  and expose an `axis_remap` config; a wrong convention silently mislocates
  every object. Calibrate against a known synthetic scene before trusting it.

## Open questions (block implementation until probed on weights)

1. **JSON box schema** — exact keys, 2D-vs-3D, metric-vs-normalized coords are
   undocumented. Probe `Cosmos3-Nano` output first; keep `_parse_boxes`
   version-tolerant.
2. **Per-box confidence** — not documented. Use a config prior until confirmed.
3. **Latency** — Cosmos is a System-2 model (seconds/clip). Sample keyframes
   (`every_n_frames`), do not call per frame. Fusion already dedups across the
   sparser stream.
4. **Camera-frame metric scale** — confirm boxes are metric meters, not
   normalized, before transforming.
5. **Pose source** — trust the caller's `pose` (SLAM/ARKit) over Cosmos's
   estimate unless benchmarked; expose both.

## Offline test plan (no GPU)

- Inject a fake `reasoner`/`transport` returning canned JSON boxes.
- Assert: parse → world-frame transform (known pose) → `Detection` with correct
  `center_xyz` / AABB; feature L2-norm; bad JSON → `AdapterError`.
- Geometry helpers (`transform_point`, `oriented_box_aabb`) unit-tested in pure
  numpy.

## Relationship to existing tracks

- Complements **path A** (`CosmosReasonVerbalizer`, shipped): A is the answer
  brain, B is the eyes. Same Cosmos family, opposite ends of the pipeline.
- An alternative B feeder is **Isaac nvblox + cuVSLAM** (metric 3D + 6-DoF
  pose, CUDA) — same `PerceptionAdapter` seam, you add labels + features.
- Slots into DEV-PLAN Phase C; does **not** block the CPU GT-adapter demo (B1').
