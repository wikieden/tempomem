"""D2 — the episodic trace lives in the core: one shared store, one id space.

`tempomem.trace.TraceLog` is the L4 episodic log (append-only sqlite, citing
scene-graph node ids). Standalone over its own path, or bound to a `TempoMem`
via `trace_log()` — the same `.smem` file, where every write first drains the
staged observations (fuse-before-persist holds across the shared connection).
Core stays numpy-only: the API takes primitives, never brain types.
"""

from __future__ import annotations

import numpy as np

from tempomem import Detection, TempoMem
from tempomem.trace import TraceLog


def _det(label: str = "mug") -> Detection:
    v = np.ones(4, dtype="float32") / 2.0
    return Detection(label, v, (0.5, 0.5, 0.8), (0.4, 0.4, 0.7), (0.6, 0.6, 0.9))


def test_standalone_record_and_query() -> None:
    log = TraceLog(":memory:")
    log.record(
        session_id="s1",
        kind="attempt",
        verb="grasp",
        subtask_id="a0",
        node_ids=(7,),
        status="success",
    )
    log.record(
        session_id="s1",
        kind="failure",
        verb="grasp",
        subtask_id="a1",
        node_ids=(7,),
        status="failed",
        error_code="ERR_GRASP_FAILED",
        detail="slipped",
        macro_parent="deliver",
    )
    rows = log.episodes(session_id="s1")
    assert [r.kind for r in rows] == ["failure", "attempt"]  # most recent first
    assert rows[0].error_code == "ERR_GRASP_FAILED"
    assert rows[0].macro_parent == "deliver"
    assert log.by_node(7) and not log.by_node(99)
    assert [r.subtask_id for r in log.failures(session_id="s1")] == ["a1"]
    log.close()


def test_trace_log_shares_the_smem_store(tmp_path) -> None:
    path = tmp_path / "t.smem"
    with TempoMem.open(path, embedding_dim=4) as mem:
        log = mem.trace_log()
        mem.add_detections([_det()])
        mem.commit()
        log.record(
            session_id="s1",
            kind="attempt",
            verb="navigate",
            subtask_id="n0",
            node_ids=(1,),
            status="success",
        )
    # one file holds both the scene graph and its episodic trace
    with TempoMem.open(path, embedding_dim=4, readonly=True) as mem2:
        assert mem2.query("mug", k=1).nodes
        assert mem2.trace_log().episodes(session_id="s1")


def test_trace_write_never_persists_unfused_observations(tmp_path) -> None:
    # fuse-before-persist across the shared connection: a trace commit while
    # observations are staged must drain them through fusion first, never
    # flush them to disk unfused
    path = tmp_path / "t.smem"
    mem = TempoMem.open(path, embedding_dim=4)
    log = mem.trace_log()
    mem.add_detections([_det()])  # staged, not yet fused
    log.record(
        session_id="s1",
        kind="attempt",
        verb="detect",
        subtask_id="d0",
        node_ids=(),
        status="success",
    )
    # the staged observation was fused by the trace write's pre-commit drain
    assert mem.query("mug", k=1).nodes
    mem.close()


def test_readonly_store_refuses_trace_writes(tmp_path) -> None:
    path = tmp_path / "t.smem"
    TempoMem.open(path, embedding_dim=4).close()
    mem = TempoMem.open(path, embedding_dim=4, readonly=True)
    log = mem.trace_log()
    try:
        log.record(
            session_id="s1",
            kind="attempt",
            verb="grasp",
            subtask_id="x",
            node_ids=(),
            status="success",
        )
        wrote = True
    except Exception:
        wrote = False
    assert not wrote
    mem.close()
