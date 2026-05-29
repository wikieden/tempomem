"""SpatialMem M0 quickstart — synthetic kitchen, no GPU, no network.

Run: python examples/01_quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from spatialmem import Detection, SpatialMemory

DIM = 32  # tiny synthetic feature dim for the demo


def _feat(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DIM).astype("float32")


def synthetic_kitchen() -> list[Detection]:
    # (label, center, seed)
    items = [
        ("mug", (1.20, 0.30, 0.94), 1),
        ("kettle", (1.55, 0.40, 0.93), 2),
        ("fridge", (2.40, 0.00, 1.20), 3),
        ("sink", (0.80, 0.35, 0.90), 4),
        ("mug", (1.22, 0.31, 0.95), 1),  # second sighting of the mug
    ]
    dets = []
    for i, (label, c, seed) in enumerate(items):
        cx, cy, cz = c
        dets.append(
            Detection(
                label=label,
                feature=_feat(seed),
                center_xyz=c,
                bbox_min=(cx - 0.05, cy - 0.05, cz - 0.05),
                bbox_max=(cx + 0.05, cy + 0.05, cz + 0.05),
                confidence=0.9,
                ts=1000.0 + i,
            )
        )
    return dets


def main() -> None:
    path = Path(tempfile.mkdtemp()) / "kitchen.smem"
    with SpatialMemory.open(path, embedding_dim=DIM) as mem:
        ids = mem.add_detections(synthetic_kitchen())
        stats = mem.commit()
        print(
            f"ingested {len(ids)} detections -> {stats.nodes_after} nodes "
            f"in {stats.elapsed_ms:.1f} ms"
        )

        hits = mem.query("mug")
        assert hits.nodes, "expected at least one hit for 'mug'"
        top = hits.nodes[0]
        print(
            f"query 'mug' -> #{top.id} '{top.label}' @ {top.center_xyz} "
            f"(intent={hits.intent_used})"
        )

        print("\n--- prompt serialization ---")
        print(mem.serialize(format="prompt"))


if __name__ == "__main__":
    main()
