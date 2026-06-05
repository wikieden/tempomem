from __future__ import annotations

import pytest

from tests.conftest import make_det


def test_moved_measures_displacement(mem) -> None:
    # same object (seed 1) seen at x=1.0 then x=1.15 -> fuses to one node;
    # moved() reports the raw first->last observation displacement (~0.15m)
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1, ts=10.0)])
    mem.commit()
    mem.add_detections([make_det("mug", (1.15, 0.0, 0.9), 1, ts=20.0)])
    mem.commit()
    assert mem.stats().n_nodes == 1
    nid = mem.recent(n=1)[0].id
    assert mem.moved(nid) == pytest.approx(0.15, abs=0.02)


def test_moved_zero_for_single_observation(mem) -> None:
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    mem.commit()
    nid = mem.recent(n=1)[0].id
    assert mem.moved(nid) == 0.0


def test_changes_new_and_seen_again(mem) -> None:
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1, ts=10.0)])
    mem.commit()
    # mug re-seen at t=100, plus a brand-new sink at t=100
    mem.add_detections([make_det("mug", (1.02, 0.0, 0.9), 1, ts=100.0)])
    mem.add_detections([make_det("sink", (5.0, 0.0, 0.9), 2, ts=100.0)])
    mem.commit()

    ch = mem.changes(since_ts=50.0)
    assert [h.label for h in ch.new] == ["sink"]  # first appeared after 50
    assert [h.label for h in ch.seen_again] == ["mug"]  # old, observed again


def test_stale_not_seen_since(mem) -> None:
    mem.add_detections([make_det("old", (0.0, 0.0, 0.0), 1, ts=10.0)])
    mem.add_detections([make_det("fresh", (5.0, 0.0, 0.0), 2, ts=100.0)])
    mem.commit()
    stale = mem.stale(before_ts=50.0)
    assert [h.label for h in stale] == ["old"]
