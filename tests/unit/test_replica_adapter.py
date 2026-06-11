"""B1\': GT Replica parsing -> world detections -> fusion. No GPU, no network.

The geometry is exercised with injected arrays (a real `.smem` Replica download
is not shipped in CI); `ReplicaFileReader`\'s PNG/traj I/O is not covered here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tempomem import SpatialMemory
from tempomem.datasets import (
    DatasetSource,
    HashEncoder,
    ReplicaAdapter,
    gt_detections_from_frame,
    stream,
)

INTR = (1.0, 1.0, 0.0, 0.0)  # fx, fy, cx, cy (cam point = (u*d, v*d, d))


def test_gt_detections_deproject_and_lift_to_world() -> None:
    depth = np.zeros((4, 4), np.float32)
    inst = np.zeros((4, 4), np.int32)
    inst[1, 1] = inst[1, 2] = 1  # mug: pixels (u,v)=(1,1),(2,1) at depth 2
    depth[1, 1] = depth[1, 2] = 2.0
    inst[3, 0] = 2  # table: pixel (u,v)=(0,3) at depth 4
    depth[3, 0] = 4.0
    dets = gt_detections_from_frame(
        depth,
        inst,
        np.eye(4),
        intrinsics=INTR,
        labels={1: "mug", 2: "table"},
        encoder=HashEncoder(16),
        min_pixels=1,
    )
    by = {d.label: d for d in dets}
    assert set(by) == {"mug", "table"}  # background id 0 dropped
    assert by["mug"].center_xyz == pytest.approx((3.0, 2.0, 2.0))  # mean of (2,2,2),(4,2,2)
    assert by["mug"].bbox_min == pytest.approx((2.0, 2.0, 2.0))
    assert by["mug"].bbox_max == pytest.approx((4.0, 2.0, 2.0))
    assert by["table"].center_xyz == pytest.approx((0.0, 12.0, 4.0))
    assert by["mug"].aux["instance_id"] == 1


def test_pose_translation_lifts_to_world() -> None:
    depth = np.zeros((2, 2), np.float32)
    inst = np.zeros((2, 2), np.int32)
    inst[0, 0] = 1
    depth[0, 0] = 3.0  # cam (0,0,3)
    pose = np.eye(4)
    pose[:3, 3] = (10.0, 0.0, -5.0)
    d = gt_detections_from_frame(
        depth, inst, pose, intrinsics=INTR, labels={1: "x"}, encoder=HashEncoder(16), min_pixels=1
    )[0]
    assert d.center_xyz == pytest.approx((10.0, 0.0, -2.0))  # (0,0,3) + (10,0,-5)


def test_replica_adapter_conforms_and_streams() -> None:
    depth = np.zeros((2, 2), np.float32)
    inst = np.zeros((2, 2), np.int32)
    inst[0, 0] = 5
    depth[0, 0] = 1.0
    ad = ReplicaAdapter(
        [(depth, inst, np.eye(4))],
        intrinsics=INTR,
        labels={5: "chair"},
        encoder=HashEncoder(16),
        min_pixels=1,
    )
    assert isinstance(ad, DatasetSource)
    frames = list(ad.frames())
    assert len(frames) == 1
    assert frames[0][0].label == "chair"


def test_replica_stream_fuses_repeated_object_to_one_node(tmp_path) -> None:
    depth = np.zeros((3, 3), np.float32)
    inst = np.zeros((3, 3), np.int32)
    inst[1, 1] = 1
    depth[1, 1] = 2.0
    enc = HashEncoder(16)
    ad = ReplicaAdapter(
        [(depth, inst, np.eye(4))] * 4,  # same GT object across 4 frames
        intrinsics=INTR,
        labels={1: "mug"},
        encoder=enc,
        min_pixels=1,
    )
    with SpatialMemory.open(tmp_path / "rep.smem", embedding_dim=16, encoder=enc) as mem:
        n_frames, n_obs = stream(mem, ad)
        assert (n_frames, n_obs) == (4, 4)
        assert mem.stats().n_nodes == 1  # 4 observations fuse to one node
