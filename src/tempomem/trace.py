"""L4 episodic trace — attempts + failures citing scene-graph node ids (D2 merge).

The memory's *time dimension*: an append-only log of what the robot tried, what
happened, and why it failed — the fuel for failure-driven replanning and the
training data loop. Every record cites scene-graph **node ids** (the shared id
space, B9), so the trace is queryable by object: "what has failed on node 7?".

Two constructions, one contract:

* ``TraceLog(path)`` — standalone log over its own sqlite file (or
  ``":memory:"``), as the brain has used since P4.
* ``TempoMem.trace_log()`` — bound to the **same `.smem` connection** as the
  scene graph (the D2 vision: one store, one id space). Every write first runs
  the store's ``before_commit`` drain, so a trace commit can never persist
  staged observations unfused (the fuse-before-persist invariant holds across
  the shared connection).

Append-only by contract: a record is **never deleted** ("clearing" a task
rotates the ``session_id``). Stdlib-only; the API takes primitives — the core
never imports brain types (dependency direction: brain → core, never back).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ._errors import StoreError

__all__ = ["TraceLog", "TraceRecord"]

# The scene store already owns an `episodes` table (perception sessions), so the
# trace rows live under their own name — required for the shared-`.smem` mode.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- 'attempt' | 'failure'
    verb        TEXT NOT NULL,
    subtask_id  TEXT NOT NULL,
    node_ids    TEXT NOT NULL,          -- JSON array of ints
    status      TEXT NOT NULL,
    error_code  TEXT,
    detail      TEXT NOT NULL DEFAULT '',
    ts          REAL NOT NULL,
    macro_parent TEXT                   -- D-D: macro verb this step expanded from
);
CREATE INDEX IF NOT EXISTS idx_trace_session ON trace_episodes(session_id);
"""


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """One append-only trace row: an attempt or a failure, citing node ids (B9)."""

    id: int
    session_id: str
    kind: str
    verb: str
    subtask_id: str
    node_ids: tuple[int, ...]
    status: str
    error_code: str | None
    detail: str
    ts: float
    macro_parent: str | None = None


class TraceLog:
    """Append-only episodic trace over sqlite (own file or a shared `.smem`).

    ``before_commit`` (used by the `TempoMem`-bound construction) runs before
    every write's commit so the owning store can drain staged work first —
    fuse-before-persist across the shared connection. ``readonly`` refuses
    writes and tolerates a store that has no trace table yet.
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        conn: sqlite3.Connection | None = None,
        before_commit: Callable[[], None] | None = None,
        readonly: bool = False,
    ) -> None:
        if (path is None) == (conn is None):
            raise ValueError("pass exactly one of path= or conn=")
        self._owns_conn = conn is None
        self._conn = conn if conn is not None else sqlite3.connect(str(path))
        self._before_commit = before_commit
        self._readonly = readonly
        if not readonly:
            self._conn.executescript(_SCHEMA)
            self._migrate_legacy()
            self._conn.commit()

    def _migrate_legacy(self) -> None:
        """Adopt rows from a pre-D2 standalone trace file (table ``episodes``).

        Only fires when the old table is trace-shaped (has ``subtask_id`` — the
        scene store's own ``episodes`` table is not) and the new table is still
        empty. Copy-only: the legacy table is left in place, never dropped.
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(episodes)")}
        if "subtask_id" not in cols:
            return
        n = self._conn.execute("SELECT COUNT(*) FROM trace_episodes").fetchone()[0]
        if n:
            return
        src_cols = "session_id, kind, verb, subtask_id, node_ids, status, error_code, detail, ts"
        macro = ", macro_parent" if "macro_parent" in cols else ", NULL"
        self._conn.execute(
            f"INSERT INTO trace_episodes ({src_cols}, macro_parent) "
            f"SELECT {src_cols}{macro} FROM episodes ORDER BY id"
        )

    def record(
        self,
        *,
        session_id: str,
        kind: str,
        verb: str,
        subtask_id: str,
        node_ids: Sequence[int] = (),
        status: str,
        error_code: str | None = None,
        detail: str = "",
        macro_parent: str | None = None,
        ts: float | None = None,
    ) -> None:
        """Append one episode row (an attempt or a failure). Never deletes.

        `ts` is seconds since the epoch (defaults to now); `node_ids` are
        scene-graph node ids in the shared B9 id space.
        """
        if self._readonly:
            raise StoreError("trace log opened read-only")
        if self._before_commit is not None:
            self._before_commit()
        self._conn.execute(
            "INSERT INTO trace_episodes "
            "(session_id, kind, verb, subtask_id, node_ids, status, error_code, detail, ts, "
            "macro_parent) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                kind,
                verb,
                subtask_id,
                json.dumps(list(node_ids)),
                status,
                error_code,
                detail,
                ts if ts is not None else time.time(),
                macro_parent,
            ),
        )
        self._conn.commit()

    def _has_table(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trace_episodes'"
        ).fetchone()
        return row is not None

    def episodes(self, *, session_id: str | None = None, limit: int = 100) -> list[TraceRecord]:
        """Recent trace records, most recent first; optionally scoped to a session."""
        if self._readonly and not self._has_table():
            return []  # a store that never traced: expected absence, not an error
        if session_id is None:
            cur = self._conn.execute(
                "SELECT * FROM trace_episodes ORDER BY id DESC LIMIT ?", (limit,)
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM trace_episodes WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
        return [self._row(r) for r in cur.fetchall()]

    def by_node(self, node_id: int, *, limit: int = 100) -> list[TraceRecord]:
        """Every record that touched `node_id` (queryable by object — B9 audit)."""
        out = self.episodes(limit=limit * 4)
        return [r for r in out if node_id in r.node_ids][:limit]

    def failures(self, *, session_id: str | None = None) -> list[TraceRecord]:
        """Just the failure records (the negation-learning view)."""
        return [r for r in self.episodes(session_id=session_id, limit=1000) if r.kind == "failure"]

    @staticmethod
    def _row(r: tuple) -> TraceRecord:
        return TraceRecord(
            id=r[0],
            session_id=r[1],
            kind=r[2],
            verb=r[3],
            subtask_id=r[4],
            node_ids=tuple(json.loads(r[5])),
            status=r[6],
            error_code=r[7],
            detail=r[8],
            ts=r[9],
            macro_parent=r[10],
        )

    def close(self) -> None:
        """Close the connection (only when this log owns it — a shared `.smem`
        connection belongs to its `TempoMem`)."""
        if self._owns_conn:
            self._conn.close()
