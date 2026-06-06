"""C3 — expose SpatialMem as LLM function-call tools (offline, no LLM/GPU).

Hand `tools.schemas()` to any function-calling LLM; route the tool calls it
emits to `tools.call(name, args)`. Returns JSON whose hits carry `node_id` so
the model can cite what it used. See the design in the `spatialmem-brain` repo.

Run: python examples/05_memory_tools.py
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np

from spatialmem import Detection, SpatialMemory, SpatialMemTools

DIM = 16


def _det(label: str, center: tuple[float, float, float], seed: int) -> Detection:
    rng = np.random.default_rng(seed)
    cx, cy, cz = center
    return Detection(
        label=label,
        feature=rng.standard_normal(DIM).astype("float32"),
        center_xyz=center,
        bbox_min=(cx - 0.05, cy - 0.05, cz - 0.05),
        bbox_max=(cx + 0.05, cy + 0.05, cz + 0.05),
        confidence=0.9,
        ts=1000.0,
    )


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp()) / "demo.smem"
    with SpatialMemory.open(tmp, embedding_dim=DIM) as mem:
        mem.add_detections(
            [
                _det("mug", (1.0, 0.0, 0.9), 1),
                _det("table", (1.0, 0.0, 0.4), 2),
                _det("chair", (2.0, 0.0, 0.5), 3),
            ]
        )
        mem.commit()
        mem.define_region("kitchen", (0.0, -1.0, 0.0), (3.0, 1.0, 2.0))

        tools = SpatialMemTools(mem)
        print("== tool schemas (hand these to your LLM) ==")
        print(json.dumps([s["name"] for s in tools.schemas()]), "\n")

        # Your LLM emits tool calls; you route them to tools.call():
        for name, args in [
            ("spatial_query", {"near": [1.0, 0.0, 0.9], "radius_m": 1.0}),
            ("whats_in", {"region": "kitchen"}),
            ("serialize_scene", {"max_tokens": 200}),
        ]:
            print(f"-> {name}({args})")
            print(json.dumps(tools.call(name, args), indent=2), "\n")


if __name__ == "__main__":
    main()
