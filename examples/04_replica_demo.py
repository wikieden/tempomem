"""04 · Replica GT stream -> fusion -> query -> viz. No GPU, no network.

Exercises the B1' `ReplicaAdapter` end to end: per-frame ground-truth instance
masks + depth + pose are deprojected to world-frame detections, streamed through
fusion (the same object across frames converges to one node), then queried.

Two modes:

    # Replica-SHAPED synthetic frames — runs anywhere, no dataset:
    uv run python examples/04_replica_demo.py -o replica_scene.smem

    # A real Replica scene via ReplicaFileReader (needs the [replica] extra):
    uv run python examples/04_replica_demo.py --scene /path/to/replica/room0

Then render the scene graph to HTML:

    uv run spatialmem viz replica_scene.smem -o replica_scene.html

NOTE on real data: a real run needs a Replica variant that ships **per-frame
instance GT masks** (e.g. the ConceptGraphs-rendered Replica, or a Habitat-sim
render of the semantic mesh). The common Nice-SLAM Replica RGB-D zip has
RGB+depth+trajectory only — no instance masks — so it cannot feed this adapter.
Supply the scene's real intrinsics and instance-id -> label map below when you
point `--scene` at it.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import numpy as np

from spatialmem import SpatialMemory
from spatialmem.bench import recall_at_k
from spatialmem.datasets import HashEncoder, ReplicaAdapter, stream

# Pinhole intrinsics (fx, fy, cx, cy) for the 16x16 synthetic frame.
SYNTH_INTRINSICS = (10.0, 10.0, 8.0, 8.0)
LABELS = {1: "mug", 2: "table", 3: "chair"}


def synthetic_replica_frames(
    n_frames: int = 12,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A few GT objects in a 16x16 instance/depth map, observed across frames
    with tiny camera jitter (poses near identity) so they fuse to one node each.
    Shaped exactly like what `ReplicaFileReader` yields for a real scene.
    """
    h = w = 16
    objects = [(1, (4, 4), 1.5), (2, (10, 8), 2.0), (3, (6, 12), 2.5)]  # id, (row, col), depth
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        depth = np.zeros((h, w), np.float32)
        instance = np.zeros((h, w), np.int32)
        for iid, (r, c), d in objects:
            instance[r - 1 : r + 1, c - 1 : c + 1] = iid  # 2x2 blob
            depth[r - 1 : r + 1, c - 1 : c + 1] = d
        pose = np.eye(4)
        pose[:3, 3] = rng.normal(0.0, 0.01, 3)  # tiny jitter -> same world pos -> fuse
        yield depth, instance, pose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", help="real Replica scene dir (needs the [replica] extra)")
    ap.add_argument("-o", "--out", default="replica_scene.smem")
    args = ap.parse_args()

    encoder = HashEncoder(512)  # match the default store / viz embedding_dim
    if args.scene:
        from spatialmem.datasets import ReplicaFileReader

        # TODO: replace with the scene's real intrinsics + instance-id->label map.
        reader = ReplicaFileReader(args.scene)
        adapter = ReplicaAdapter(
            reader, intrinsics=SYNTH_INTRINSICS, labels=LABELS, encoder=encoder, min_pixels=50
        )
    else:
        adapter = ReplicaAdapter(
            list(synthetic_replica_frames()),
            intrinsics=SYNTH_INTRINSICS,
            labels=LABELS,
            encoder=encoder,
            min_pixels=1,
        )

    with SpatialMemory.open(args.out, embedding_dim=512, encoder=encoder) as mem:
        n_frames, n_obs = stream(mem, adapter)
        report = recall_at_k(mem, [("mug", "mug"), ("table", "table"), ("chair", "chair")], k=5)
        print(f"streamed {n_frames} frames, {n_obs} observations -> {mem.stats().n_nodes} nodes")
        print(f"recall@5 = {report.recall:.2f}  (misses: {report.misses})")

    print(f"wrote {args.out}")
    print(f"viz:  uv run spatialmem viz {args.out} -o replica_scene.html")


if __name__ == "__main__":
    main()
