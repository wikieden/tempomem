"""SpatialMem Phase-B demo — stream a multi-frame scene, then query it.

No GPU, no download. A SyntheticScene stands in for a Replica/ScanNet stream:
each object is observed across many frames, and incremental fusion converges
the observations to one node per object — the core pitch. Then we query and
answer over the resulting graph.

Run: python examples/03_stream_scene.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from spatialmem import HashEncoder, SpatialMemory, SyntheticScene, stream

DIM = 64

OBJECTS = [
    ("mug", (1.20, 0.0, 0.94)),
    ("kettle", (1.55, 0.0, 0.93)),
    ("fridge", (3.40, 0.0, 1.20)),
    ("sink", (0.80, 0.0, 0.90)),
    ("microwave", (2.40, 0.0, 1.05)),
]

QUESTIONS = ["mug", "kettle", "fridge", "sink", "microwave"]


class EchoVerbalizer:
    def complete(self, prompt: str) -> str:
        for line in prompt.splitlines():
            if line.strip().startswith("#"):
                return f"Found: {line.strip()}"
        return "Not in this scene."


def main() -> None:
    enc = HashEncoder(DIM)
    scene = SyntheticScene(objects=list(OBJECTS), encoder=enc, n_frames=15)
    path = Path(tempfile.mkdtemp()) / "kitchen.smem"

    with SpatialMemory.open(
        path, embedding_dim=DIM, encoder=enc, verbalizer=EchoVerbalizer()
    ) as mem:
        n_frames, n_obs = stream(mem, scene, commit_every=5)
        st = mem.stats()
        print(f"streamed {n_frames} frames, {n_obs} observations")
        print(f"after fusion: {st.n_nodes} nodes  (dedup ratio {n_obs / max(st.n_nodes, 1):.1f}x)")

        hits = 0
        for q in QUESTIONS:
            res = mem.query(q, k=3)
            ok = bool(res.nodes) and res.nodes[0].label == q
            hits += int(ok)
            top = res.nodes[0].label if res.nodes else "—"
            print(f"  Q '{q}' -> {top} {'OK' if ok else 'MISS'}")
        print(f"recall: {hits}/{len(QUESTIONS)}")

        print("\nanswer('mug') ->", mem.answer("mug"))


if __name__ == "__main__":
    main()
