from __future__ import annotations

import numpy as np
import pytest

from tempomem import Detection, IngestError, PerceptionAdapter, SpatialMemory
from tests.conftest import DIM


class StubAdapter:
    """Fake perception: ignores pixels, emits two fixed detections per frame."""

    def process_frame(self, rgb, depth, pose, intrinsics=None) -> list[Detection]:
        def det(label, c):
            cx, cy, cz = c
            return Detection(
                label=label,
                feature=np.ones(DIM, dtype="float32"),
                center_xyz=c,
                bbox_min=(cx - 0.05, cy - 0.05, cz - 0.05),
                bbox_max=(cx + 0.05, cy + 0.05, cz + 0.05),
                confidence=0.9,
                ts=float(pose[0, 3]),  # use pose tx as a fake timestamp
            )

        return [det("mug", (1.0, 0.0, 0.0)), det("sink", (3.0, 0.0, 0.0))]


def _frame(tx: float):
    rgb = np.zeros((4, 4, 3), dtype="uint8")
    depth = np.ones((4, 4), dtype="float32")
    pose = np.eye(4, dtype="float32")
    pose[0, 3] = tx
    return rgb, depth, pose


def test_stub_adapter_satisfies_protocol() -> None:
    assert isinstance(StubAdapter(), PerceptionAdapter)


def test_add_frame_ingests_detections(tmp_path) -> None:
    with SpatialMemory.open(tmp_path / "f.smem", embedding_dim=DIM, adapter=StubAdapter()) as mem:
        rgb, depth, pose = _frame(10.0)
        ids = mem.add_frame(rgb, depth, pose)
        assert len(ids) == 2
        mem.commit()
        labels = {n.label for n in mem.recent(n=10)}
        assert labels == {"mug", "sink"}


def test_add_frame_adapter_per_call(tmp_path) -> None:
    with SpatialMemory.open(tmp_path / "f2.smem", embedding_dim=DIM) as mem:
        rgb, depth, pose = _frame(0.0)
        mem.add_frame(rgb, depth, pose, adapter=StubAdapter())
        mem.commit()
        assert mem.stats().n_nodes == 2


def test_add_frame_without_adapter_raises(tmp_path) -> None:
    with SpatialMemory.open(tmp_path / "f3.smem", embedding_dim=DIM) as mem:
        rgb, depth, pose = _frame(0.0)
        with pytest.raises(IngestError):
            mem.add_frame(rgb, depth, pose)
