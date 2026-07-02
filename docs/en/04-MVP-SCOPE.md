> 🌐 **English** · [中文](../zh/04-MVP-SCOPE.md)

# 04 · MVP Scope (M1)

## Goal

Ship the smallest thing a real user can `pip install` and immediately get value from. Defer perception. Win the API.

## In Scope

1. **Ingest:** `add_detections(List[Detection])` only. `Detection` = `{label, conf, center_xyz, bbox3d, feature_vec, mask=None, ts}`.
2. **Fusion:** deterministic arbiter — KNN candidate (centroid distance ≤ τ_d), 3D IoU ≥ τ_iou, CLIP-cos ≥ τ_s, label compatibility check.
3. **Store:** SQLite + sqlite-vec + R-tree. Single `.smem` file.
4. **Query:**
   - `mem.query(text)` → routes to spatial / semantic / temporal retriever, returns ranked nodes (no LLM yet — pure retrieval).
   - `mem.spatial(near=(x,y,z), radius=r)` → R-tree range scan.
   - `mem.semantic(text)` → CLIP-text embed + ANN over node features.
   - `mem.recent(n=10)` → temporal scan.
5. **Serialize:** `mem.serialize(format="prompt", k_hops=2, root=None)` → token-budgeted indented text.
6. **CLI:** `tempomem inspect demo.smem` (counts, sample nodes, schema version).

## Out of Scope (MVP)

- Real RGB-D perception (M2)
- LLM verbalizer (M2 — MVP returns raw nodes; user wraps in their own LLM call)
- ROS 2 bridge (M3)
- Web viewer (stretch — only if time)
- Decay / forget (M2)
- gRPC (M4)
- Multi-room hierarchy inference (M2)

## Demo Script (recorded for launch)

```python
import json, numpy as np
from tempomem import TempoMem, Detection

mem = TempoMem.open("kitchen.smem")

# Simulate 3 passes through a kitchen
for det in load_synthetic_kitchen_detections():
    mem.add_detections([det])
mem.commit()

# Query
hits = mem.query("mug near the sink")
print(hits[0].label, hits[0].center_xyz, hits[0].confidence)

# Prompt-ready text for any LLM
print(mem.serialize(format="prompt", k_hops=1, root=hits[0].id))
```

Runs in <30 s on a laptop, no GPU, no network.

## Acceptance Tests (must all pass before tagging v0.1.0)

| ID | Check | Tool |
|---|---|---|
| A1 | `pip install tempomem` on macOS arm64 + Linux x86_64 (Python 3.10/3.11/3.12) | CI matrix |
| A2 | Quickstart in README runs without edits | `pytest tests/test_readme.py` |
| A3 | 10k detection insert + 100 queries finishes in <60 s on a 2024 MacBook Air | benchmark gate |
| A4 | `.smem` file produced by v0.1.0 readable by v0.1.x — schema migrations covered | migration test |
| A5 | `mem.stats()` numbers match a recomputed-from-scratch baseline | invariant test |
| A6 | Coverage ≥ 75% on `fusion`, `store`, `query` modules | `pytest --cov` |
| A7 | No hard dep on Torch / CUDA / ROS in default install | `pip show tempomem` deps audit |

## Risk Register (MVP)

| Risk | Severity | Mitigation |
|---|---|---|
| Arbiter merges things it shouldn't | high | Deterministic + thresholds in config; gold-set regression tests; logged decisions |
| sqlite-vec ANN recall low at small N | medium | Hybrid: linear scan under 10k vectors; switch to ANN above |
| API churn after v0.1.0 lock-in | high | Keep API surface tiny; mark non-core as `experimental` |
| ConceptGraphs install pain leaks into core | high | Hard rule: ConceptGraphs lives behind `pip install tempomem[conceptgraphs]` |
