from __future__ import annotations

from tests.conftest import make_det


def test_consolidate_merges_near_duplicate(mem) -> None:
    # two sightings of the same mug 0.36 m apart: far enough that fusion's
    # candidate gate misses them at ingest (two nodes), close enough that
    # consolidate's pairwise score clears tau_merge.
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    mem.commit()
    mem.add_detections([make_det("mug", (1.36, 0.0, 0.9), 1)])
    mem.commit()
    assert mem.stats().n_nodes == 2
    merged = mem.consolidate()
    assert merged >= 1
    assert mem.stats().n_nodes == 1  # collapsed to one mug


def test_consolidate_keeps_distinct_objects(mem) -> None:
    mem.add_detections([make_det("mug", (0.0, 0.0, 0.0), 1), make_det("sink", (5.0, 0.0, 0.0), 2)])
    mem.commit()
    assert mem.consolidate() == 0
    assert mem.stats().n_nodes == 2


def test_salient_ranks_recent_and_confident(mem) -> None:
    mem.add_detections([make_det("old", (0.0, 0.0, 0.0), 1, ts=1.0)])
    mem.add_detections([make_det("new", (5.0, 0.0, 0.0), 2, ts=100.0)])
    mem.commit()
    top = mem.salient(n=2)
    assert top[0].label == "new"  # more recent → more salient
    assert top[0].score >= top[1].score


def test_salient_empty(mem) -> None:
    assert mem.salient() == []
