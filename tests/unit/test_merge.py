from __future__ import annotations

import pytest

from spatialmem import SchemaMismatchError, SpatialMemory, StoreError
from tests.conftest import DIM, make_det


def _store(tmp_path, name: str, dim: int = DIM) -> SpatialMemory:
    return SpatialMemory.open(tmp_path / name, embedding_dim=dim)


def test_merge_dedups_shared_object_and_adds_new(tmp_path) -> None:
    a = _store(tmp_path, "a.smem")
    a.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    a.commit()

    b = _store(tmp_path, "b.smem")
    b.add_detections([make_det("mug", (1.02, 0.0, 0.9), 1), make_det("sink", (0.0, 1.0, 0.9), 2)])
    b.commit()
    b.close()

    stats = a.merge(tmp_path / "b.smem")
    assert stats.observations_committed == 2  # 2 source objects fed through fusion
    # shared mug deduped, new sink added -> 2 nodes
    assert a.stats().n_nodes == 2
    assert {h.label for h in a.recent(n=10)} == {"mug", "sink"}
    a.close()


def test_merge_dim_mismatch(tmp_path) -> None:
    a = _store(tmp_path, "a.smem")
    a.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    a.commit()
    b = SpatialMemory.open(tmp_path / "b.smem", embedding_dim=DIM + 1)
    b.close()
    with pytest.raises(SchemaMismatchError):
        a.merge(tmp_path / "b.smem")
    a.close()


def test_merge_readonly_rejected(tmp_path) -> None:
    a = _store(tmp_path, "a.smem")
    a.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    a.commit()
    a.close()
    ro = SpatialMemory.open(tmp_path / "a.smem", embedding_dim=DIM, readonly=True)
    with pytest.raises(StoreError):
        ro.merge(tmp_path / "a.smem")
    ro.close()
