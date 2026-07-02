"""Fuse-before-persist invariant: maintenance never flushes un-fused observations.

`add_detections()` stages observation rows in `self._pending` that are not yet
linked to a node — fusion is deferred to `commit()`. A maintenance method that
issues its own `conn.commit()` between `add_detections()` and `commit()` (the
cognitive-tick pattern `add; consolidate; decay; commit`) used to flush those
rows to disk unfused, leaving orphan observations on a crash and showing a
half-ingested store to whatever ran in between. Regression for that bug.
"""

from __future__ import annotations

import sqlite3

from tempomem import TempoMem
from tests.conftest import DIM, make_det


def _orphan_count(conn: sqlite3.Connection) -> int:
    """Observation rows not linked to any node via node_obs."""
    return int(
        conn.execute(
            "SELECT COUNT(*) AS c FROM observations o "
            "WHERE NOT EXISTS (SELECT 1 FROM node_obs n WHERE n.obs_id = o.id)"
        ).fetchone()["c"]
    )


def test_decay_between_add_and_commit_fuses_first(mem) -> None:
    # (a) add observations, run maintenance, then commit. The maintenance commit
    # must fuse the staged observation first, not flush it as an orphan.
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    mem.decay(now=1000.0)  # now == obs ts → no age-decay, just exercises the commit
    assert mem.stats().n_nodes == 1  # staged obs was fused, not left dangling
    nodes = mem.recent(n=10)
    assert len(mem.history(nodes[0].id)) == 1  # observation linked to the node
    # the user's own commit is a no-op for the already-drained obs
    cs = mem.commit()
    assert cs.observations_committed == 0
    assert mem.stats().n_nodes == 1


def test_consolidate_between_add_and_commit_keeps_linkage(mem) -> None:
    # (a, variant) consolidate interleaved before commit must still leave every
    # observation linked to a node.
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1), make_det("sink", (5.0, 0.0, 0.0), 2)])
    mem.consolidate()
    assert mem.stats().n_nodes == 2
    assert _orphan_count(mem._conn) == 0
    mem.commit()
    assert _orphan_count(mem._conn) == 0


def test_no_committed_orphans_on_disk_after_interleaved_maintenance(tmp_path) -> None:
    # (b) simulate "crash + inspect": a second read-only connection sees only
    # committed-to-disk state. After an interleaved maintenance commit there must
    # be no observation persisted without a node link.
    path = tmp_path / "tick.smem"
    m = TempoMem.open(path, embedding_dim=DIM)
    try:
        m.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        m.consolidate()  # stray maintenance commit between add and commit
        reader = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        reader.row_factory = sqlite3.Row
        try:
            assert _orphan_count(reader) == 0
        finally:
            reader.close()
    finally:
        m.close()


def test_close_fuses_pending_no_orphans(tmp_path) -> None:
    # close() commits unconditionally too — forgetting commit() before close must
    # not persist orphan observations.
    path = tmp_path / "tick2.smem"
    with TempoMem.open(path, embedding_dim=DIM) as m:
        m.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        # no explicit commit; __exit__ → close() must fuse first
    reader = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    reader.row_factory = sqlite3.Row
    try:
        assert _orphan_count(reader) == 0
        assert int(reader.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]) == 1
    finally:
        reader.close()
