from __future__ import annotations

import numpy as np

from tempomem import Detection, SpatialMemory
from tempomem.fusion import iou3d, label_compat
from tests.conftest import DIM, make_det


def test_iou3d_identical() -> None:
    assert iou3d((0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)) == 1.0


def test_iou3d_disjoint() -> None:
    assert iou3d((0, 0, 0), (1, 1, 1), (5, 5, 5), (6, 6, 6)) == 0.0


def test_label_compat_exact_and_miss() -> None:
    assert label_compat("mug", [("mug", 0.9)]) >= 0.8
    assert label_compat("mug", [("fridge", 1.0)]) == 0.0
    assert label_compat("cup", [("coffee cup", 1.0)]) == 0.5  # substring


def test_dedup_same_object_merges(mem) -> None:
    # same label + same position + same feature seed -> one node, n_obs=2
    mem.add_detections([make_det("mug", (1.0, 0.3, 0.9), 1, ts=10.0)])
    mem.add_detections([make_det("mug", (1.01, 0.31, 0.9), 1, ts=11.0)])
    mem.commit()
    st = mem.stats()
    assert st.n_nodes == 1
    assert st.n_obs == 2
    node = mem.recent(n=1)[0]
    assert node.label == "mug"


def test_distinct_objects_stay_separate(mem) -> None:
    mem.add_detections([make_det("mug", (1.0, 0.3, 0.9), 1)])
    mem.add_detections([make_det("fridge", (5.0, 0.0, 1.2), 2)])
    mem.commit()
    assert mem.stats().n_nodes == 2


def test_reject_low_confidence(tmp_path) -> None:
    m = SpatialMemory.open(tmp_path / "r.smem", embedding_dim=DIM)
    rng = np.random.default_rng(7)
    low = Detection(
        "ghost",
        rng.standard_normal(DIM).astype("float32"),
        (0, 0, 0),
        (-0.05, -0.05, -0.05),
        (0.05, 0.05, 0.05),
        confidence=0.1,  # below tau_obs=0.30
    )
    m.add_detections([low])
    m.commit()
    assert m.stats().n_nodes == 0
    m.close()


def test_determinism(tmp_path) -> None:
    stream = [
        make_det("mug", (1.0, 0.3, 0.9), 1, ts=1.0),
        make_det("kettle", (1.5, 0.4, 0.9), 2, ts=2.0),
        make_det("mug", (1.02, 0.29, 0.9), 1, ts=3.0),
        make_det("fridge", (5.0, 0.0, 1.2), 3, ts=4.0),
    ]

    def build(path) -> list[tuple]:
        m = SpatialMemory.open(path, embedding_dim=DIM)
        for d in stream:
            m.add_detections([d])
        m.commit()
        nodes = sorted((h.label, round(h.center_xyz[0], 6)) for h in m.recent(n=100))
        m.close()
        return nodes

    a = build(tmp_path / "a.smem")
    b = build(tmp_path / "b.smem")
    assert a == b
    assert len(a) == 3  # mug merged, kettle + fridge distinct
