"""Unit tests for SpatialMemConfig.max_pending_obs auto-flush behaviour.

When max_pending_obs is set, add_detections() must:
  1. Call commit() automatically once _pending reaches the threshold.
  2. Emit a WARNING log line via the spatialmem.memory logger.
  3. Leave _pending empty after the auto-flush.
  4. Produce a fully fused store (no orphan observations).

When max_pending_obs is None (default) the behaviour is unchanged —
_pending may grow without bound until the caller calls commit().
"""

from __future__ import annotations

import logging

import pytest

from spatialmem import SpatialMemConfig, SpatialMemory
from tests.conftest import DIM, make_det

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _orphan_count(mem: SpatialMemory) -> int:
    return int(
        mem._conn.execute(
            "SELECT COUNT(*) AS c FROM observations o "
            "WHERE NOT EXISTS (SELECT 1 FROM node_obs n WHERE n.obs_id = o.id)"
        ).fetchone()["c"]
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_auto_flush_triggers_at_threshold(tmp_path) -> None:
    """_pending is drained once its length hits max_pending_obs."""
    cfg = SpatialMemConfig(max_pending_obs=3)
    with SpatialMemory.open(tmp_path / "ap.smem", embedding_dim=DIM, config=cfg) as mem:
        # add 2 detections — below threshold, _pending should still hold them
        mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        mem.add_detections([make_det("sink", (5.0, 0.0, 0.0), 2)])
        assert len(mem._pending) == 2

        # adding the 3rd detection crosses the threshold → auto-flush
        mem.add_detections([make_det("chair", (3.0, 0.0, 1.0), 3)])
        assert len(mem._pending) == 0  # flushed
        assert mem.stats().n_nodes == 3
        assert _orphan_count(mem) == 0


def test_auto_flush_emits_warning(tmp_path, caplog) -> None:
    """A WARNING is logged through the spatialmem.memory logger on auto-flush."""
    cfg = SpatialMemConfig(max_pending_obs=2)
    with (
        caplog.at_level(logging.WARNING, logger="spatialmem.memory"),
        SpatialMemory.open(tmp_path / "warn.smem", embedding_dim=DIM, config=cfg) as mem,
    ):
        mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        assert len(caplog.records) == 0  # not yet at threshold

        mem.add_detections([make_det("sink", (5.0, 0.0, 0.0), 2)])
        # threshold reached — warning must have been emitted
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert rec.levelno == logging.WARNING
        assert "auto-flushing" in rec.getMessage()
        assert "max_pending_obs=2" in rec.getMessage()


def test_no_auto_flush_when_limit_is_none(tmp_path) -> None:
    """Default config (max_pending_obs=None) never auto-flushes."""
    with SpatialMemory.open(tmp_path / "no_af.smem", embedding_dim=DIM) as mem:
        for i in range(10):
            mem.add_detections([make_det(f"obj{i}", (float(i), 0.0, 0.9), i + 1)])
        # all 10 still pending — no auto-flush
        assert len(mem._pending) == 10
        mem.commit()
        assert len(mem._pending) == 0


def test_auto_flush_batch_add_detections(tmp_path) -> None:
    """Threshold applies to cumulative _pending length, not per-call batch size."""
    cfg = SpatialMemConfig(max_pending_obs=3)
    with SpatialMemory.open(tmp_path / "batch.smem", embedding_dim=DIM, config=cfg) as mem:
        # single call with a batch of 5 — exceeds threshold, must auto-flush
        mem.add_detections(
            [
                make_det("a", (1.0, 0.0, 0.9), 1),
                make_det("b", (2.0, 0.0, 0.9), 2),
                make_det("c", (3.0, 0.0, 0.9), 3),
                make_det("d", (4.0, 0.0, 0.9), 4),
                make_det("e", (5.0, 0.0, 0.9), 5),
            ]
        )
        assert len(mem._pending) == 0
        assert mem.stats().n_nodes == 5
        assert _orphan_count(mem) == 0


def test_explicit_commit_after_auto_flush_is_noop(tmp_path) -> None:
    """An explicit commit() after an auto-flush commits 0 observations."""
    cfg = SpatialMemConfig(max_pending_obs=2)
    with SpatialMemory.open(tmp_path / "noop.smem", embedding_dim=DIM, config=cfg) as mem:
        mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        mem.add_detections([make_det("sink", (5.0, 0.0, 0.0), 2)])
        # auto-flush has already fired; explicit commit should be a no-op
        stats = mem.commit()
        assert stats.observations_committed == 0


def test_max_pending_obs_below_one_rejected() -> None:
    """A threshold < 1 is a config mistake and must raise at construction."""
    with pytest.raises(ValueError, match="max_pending_obs"):
        SpatialMemConfig(max_pending_obs=0)
    with pytest.raises(ValueError, match="max_pending_obs"):
        SpatialMemConfig(max_pending_obs=-3)


def test_auto_flush_rearms_after_each_threshold(tmp_path) -> None:
    """Crossing the threshold a second time flushes again (the guard re-arms)."""
    cfg = SpatialMemConfig(max_pending_obs=2)
    with SpatialMemory.open(tmp_path / "rearm.smem", embedding_dim=DIM, config=cfg) as mem:
        mem.add_detections([make_det("a", (1.0, 0.0, 0.9), 1)])
        mem.add_detections([make_det("b", (2.0, 0.0, 0.9), 2)])  # flush #1
        assert len(mem._pending) == 0
        mem.add_detections([make_det("c", (3.0, 0.0, 0.9), 3)])
        assert len(mem._pending) == 1  # re-armed, below threshold again
        mem.add_detections([make_det("d", (4.0, 0.0, 0.9), 4)])  # flush #2
        assert len(mem._pending) == 0
        assert mem.stats().n_nodes == 4
        assert _orphan_count(mem) == 0


def test_auto_flush_preserves_distinct_episodes(tmp_path) -> None:
    """Observations staged under different episode= survive the flush with their binding."""
    cfg = SpatialMemConfig(max_pending_obs=2)
    with SpatialMemory.open(tmp_path / "ep.smem", embedding_dim=DIM, config=cfg) as mem:
        mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)], episode="a")
        mem.add_detections([make_det("sink", (5.0, 0.0, 0.0), 2)], episode="b")  # flush
        assert len(mem._pending) == 0
        sessions = {r["session"] for r in mem._conn.execute("SELECT session FROM episodes")}
        assert {"a", "b"} <= sessions
        assert _orphan_count(mem) == 0
