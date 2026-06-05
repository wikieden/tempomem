from __future__ import annotations

import pytest

from spatialmem import StoreError
from tests.conftest import make_det


def _one_node(mem) -> int:
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    mem.commit()
    return mem.recent(n=1)[0].id


def test_update_label(mem) -> None:
    nid = _one_node(mem)
    mem.update(nid, label="coffee mug")
    assert mem.recent(n=1)[0].label == "coffee mug"


def test_update_moves_centroid_and_bbox(mem) -> None:
    nid = _one_node(mem)
    mem.update(nid, center_xyz=(5.0, 0.0, 0.9))
    hit = mem.recent(n=1)[0]
    assert hit.center_xyz[0] == pytest.approx(5.0)
    # bbox keeps its extent, recentred
    res = mem.spatial(near=(5.0, 0.0, 0.9), radius=0.2)
    assert res and res[0].id == nid


def test_update_confidence_bounds(mem) -> None:
    nid = _one_node(mem)
    with pytest.raises(StoreError):
        mem.update(nid, confidence=1.5)


def test_update_missing_node(mem) -> None:
    with pytest.raises(StoreError):
        mem.update(999, label="x")


def test_history_is_observation_trail(mem) -> None:
    # same object seen three frames at slightly different spots -> one node, 3 obs
    mem.add_detections([make_det("mug", (1.00, 0.0, 0.9), 1, ts=10.0)])
    mem.commit()
    mem.add_detections([make_det("mug", (1.02, 0.0, 0.9), 1, ts=20.0)])
    mem.commit()
    mem.add_detections([make_det("mug", (1.04, 0.0, 0.9), 1, ts=30.0)])
    mem.commit()
    nid = mem.recent(n=1)[0].id
    trail = mem.history(nid)
    assert len(trail) == 3
    assert [o.ts for o in trail] == [10.0, 20.0, 30.0]
    assert trail[-1].center_xyz[0] == pytest.approx(1.04)  # last seen position
