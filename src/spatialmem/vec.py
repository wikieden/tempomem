"""Optional sqlite-vec ANN index over node features (the `[vec]` extra).

A `node_vec` vec0 virtual table mirrors `nodes.feature`, maintained on
insert/update/delete. When the extension is unavailable the table is never
created and callers fall back to a linear cosine scan. The BLOB feature in
`nodes` stays the source of truth; `node_vec` is a rebuildable index.

distance_metric=cosine → vec0 distance is cosine distance (1 - cos), so a
retrieval score is simply `1 - distance`.
"""

from __future__ import annotations

import sqlite3

import numpy as np


def _load_extension(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError):  # pragma: no cover - platform dependent
        return False
    return True


def enabled(conn: sqlite3.Connection) -> bool:
    """True if the node_vec index table exists in this store."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='node_vec'"
        ).fetchone()
        is not None
    )


def _serialize(feature: np.ndarray) -> bytes:
    import sqlite_vec

    return sqlite_vec.serialize_float32(np.asarray(feature, dtype=np.float32).reshape(-1).tolist())


def try_enable(conn: sqlite3.Connection, embedding_dim: int, readonly: bool) -> bool:
    """Load the extension and ensure the index exists. Returns True if active."""
    if not _load_extension(conn):
        return False
    if readonly:
        return enabled(conn)
    dim = int(embedding_dim)  # validated int — interpolated into the vec0 DDL
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS node_vec USING vec0(emb float[{dim}] "
        f"distance_metric=cosine)"
    )
    _backfill(conn)
    return True


def _backfill(conn: sqlite3.Connection) -> None:
    have = {int(r[0]) for r in conn.execute("SELECT rowid FROM node_vec")}
    for r in conn.execute("SELECT id, feature FROM nodes"):
        nid = int(r["id"])
        if nid not in have:
            conn.execute(
                "INSERT INTO node_vec(rowid, emb) VALUES(?, ?)",
                (nid, _serialize(np.frombuffer(r["feature"], dtype=np.float32))),
            )


def upsert(conn: sqlite3.Connection, node_id: int, feature: np.ndarray) -> None:
    if not enabled(conn):
        return
    conn.execute("DELETE FROM node_vec WHERE rowid=?", (node_id,))
    conn.execute("INSERT INTO node_vec(rowid, emb) VALUES(?, ?)", (node_id, _serialize(feature)))


def delete(conn: sqlite3.Connection, node_id: int) -> None:
    if not enabled(conn):
        return
    conn.execute("DELETE FROM node_vec WHERE rowid=?", (node_id,))


def search(conn: sqlite3.Connection, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(node_id, score)] by cosine, best first. score = 1 - cosine_distance."""
    rows = conn.execute(
        "SELECT rowid, distance FROM node_vec WHERE emb MATCH ? ORDER BY distance LIMIT ?",
        (_serialize(qvec), k),
    ).fetchall()
    return [(int(r["rowid"]), max(0.0, 1.0 - float(r["distance"]))) for r in rows]
