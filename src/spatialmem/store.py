"""Graph store: row-level CRUD over the SQLite connection + stats.

Higher-level fusion/query build on these primitives. See spec/SCHEMA.md.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import vec as _vec

Vec3 = tuple[float, float, float]


@dataclass
class StoreStats:
    n_nodes: int
    n_edges: int
    n_obs: int
    n_episodes: int
    store_bytes: int


@dataclass
class NodeRow:
    id: int
    type: str
    label: str
    labels: list[tuple[str, float]]
    confidence: float
    centroid: Vec3
    bbox_min: Vec3
    bbox_max: Vec3
    n_obs: int
    t_first: float
    t_last: float
    parent_id: int | None


def _vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def ensure_episode(conn: sqlite3.Connection, session: str, ts: float) -> int:
    row = conn.execute(
        "SELECT id FROM episodes WHERE session=? ORDER BY id DESC LIMIT 1", (session,)
    ).fetchone()
    if row is not None:
        return int(row["id"])
    cur = conn.execute("INSERT INTO episodes(session, start_ts) VALUES(?, ?)", (session, ts))
    return int(cur.lastrowid)


def insert_observation(
    conn: sqlite3.Connection,
    *,
    episode_id: int,
    ts: float,
    label: str,
    confidence: float,
    center: Vec3,
    bbox_min: Vec3,
    bbox_max: Vec3,
    feature: np.ndarray,
    mask_rle: bytes | None,
    aux: dict | None,
) -> int:
    cur = conn.execute(
        """INSERT INTO observations(
            episode_id, ts, label, confidence,
            center_x, center_y, center_z,
            bbox_min_x, bbox_min_y, bbox_min_z,
            bbox_max_x, bbox_max_y, bbox_max_z,
            feature, mask_rle, aux)
           VALUES(?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?)""",
        (
            episode_id,
            ts,
            label,
            confidence,
            center[0],
            center[1],
            center[2],
            bbox_min[0],
            bbox_min[1],
            bbox_min[2],
            bbox_max[0],
            bbox_max[1],
            bbox_max[2],
            _vec_to_blob(feature),
            mask_rle,
            json.dumps(aux) if aux else None,
        ),
    )
    return int(cur.lastrowid)


def insert_node(
    conn: sqlite3.Connection,
    *,
    type_: str,
    label: str,
    labels: list[tuple[str, float]],
    confidence: float,
    centroid: Vec3,
    bbox_min: Vec3,
    bbox_max: Vec3,
    feature: np.ndarray,
    n_obs: int,
    t_first: float,
    t_last: float,
    parent_id: int | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO nodes(
            type, label, labels_json, confidence,
            centroid_x, centroid_y, centroid_z,
            bbox_min_x, bbox_min_y, bbox_min_z,
            bbox_max_x, bbox_max_y, bbox_max_z,
            feature, n_obs, t_first, t_last, parent_id)
           VALUES(?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?,?,?)""",
        (
            type_,
            label,
            json.dumps(labels),
            confidence,
            centroid[0],
            centroid[1],
            centroid[2],
            bbox_min[0],
            bbox_min[1],
            bbox_min[2],
            bbox_max[0],
            bbox_max[1],
            bbox_max[2],
            _vec_to_blob(feature),
            n_obs,
            t_first,
            t_last,
            parent_id,
        ),
    )
    node_id = int(cur.lastrowid)
    _vec.upsert(conn, node_id, feature)
    return node_id


def link_node_obs(conn: sqlite3.Connection, node_id: int, obs_id: int, ts: float) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO node_obs(node_id, obs_id, ts) VALUES(?,?,?)",
        (node_id, obs_id, ts),
    )


def upsert_edge(
    conn: sqlite3.Connection, src: int, dst: int, type_: str, confidence: float, t_last: float
) -> None:
    conn.execute(
        """INSERT INTO edges(src, dst, type, confidence, t_last)
           VALUES(?,?,?,?,?)
           ON CONFLICT(src, dst, type)
           DO UPDATE SET confidence=excluded.confidence, t_last=excluded.t_last""",
        (src, dst, type_, confidence, t_last),
    )


def _row_to_node(r: sqlite3.Row) -> NodeRow:
    return NodeRow(
        id=int(r["id"]),
        type=r["type"],
        label=r["label"],
        labels=[tuple(x) for x in json.loads(r["labels_json"])],
        confidence=float(r["confidence"]),
        centroid=(r["centroid_x"], r["centroid_y"], r["centroid_z"]),
        bbox_min=(r["bbox_min_x"], r["bbox_min_y"], r["bbox_min_z"]),
        bbox_max=(r["bbox_max_x"], r["bbox_max_y"], r["bbox_max_z"]),
        n_obs=int(r["n_obs"]),
        t_first=float(r["t_first"]),
        t_last=float(r["t_last"]),
        parent_id=int(r["parent_id"]) if r["parent_id"] is not None else None,
    )


def get_node(conn: sqlite3.Connection, node_id: int) -> NodeRow | None:
    r = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
    return _row_to_node(r) if r else None


def node_feature(conn: sqlite3.Connection, node_id: int) -> np.ndarray | None:
    r = conn.execute("SELECT feature FROM nodes WHERE id=?", (node_id,)).fetchone()
    return _blob_to_vec(r["feature"]) if r else None


def all_nodes(conn: sqlite3.Connection) -> list[NodeRow]:
    return [_row_to_node(r) for r in conn.execute("SELECT * FROM nodes ORDER BY id")]


def recent_nodes(conn: sqlite3.Connection, n: int) -> list[NodeRow]:
    rows = conn.execute("SELECT * FROM nodes ORDER BY t_last DESC LIMIT ?", (n,))
    return [_row_to_node(r) for r in rows]


def delete_node(conn: sqlite3.Connection, node_id: int) -> None:
    conn.execute("DELETE FROM node_obs WHERE node_id=?", (node_id,))
    conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (node_id, node_id))
    conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
    _vec.delete(conn, node_id)


def set_confidence(conn: sqlite3.Connection, node_id: int, confidence: float) -> None:
    conn.execute("UPDATE nodes SET confidence=? WHERE id=?", (confidence, node_id))


def decay_and_prune(
    conn: sqlite3.Connection, *, now: float, half_life_days: float, min_conf: float
) -> tuple[int, int]:
    """Age-decay node confidence and prune below a floor.

    conf' = conf * 0.5 ** (age_days / half_life_days), age from t_last.
    Returns (n_decayed, n_pruned). Pure read+write, caller commits.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    decayed = 0
    pruned = 0
    for n in all_nodes(conn):
        age_days = max(0.0, (now - n.t_last) / 86400.0)
        factor = 0.5 ** (age_days / half_life_days)
        new_conf = n.confidence * factor
        if new_conf < min_conf:
            delete_node(conn, n.id)
            pruned += 1
        elif factor < 1.0:
            set_confidence(conn, n.id, new_conf)
            decayed += 1
    return decayed, pruned


def _aabb_overlap(amin: Vec3, amax: Vec3, bmin: Vec3, bmax: Vec3, dilation: float) -> bool:
    return all(not (amax[i] + dilation < bmin[i] or bmax[i] + dilation < amin[i]) for i in range(3))


def candidates_near(
    conn: sqlite3.Connection, bbox_min: Vec3, bbox_max: Vec3, dilation: float
) -> list[NodeRow]:
    """Nodes whose bbox overlaps the observation bbox dilated by `dilation`.

    M0/M1 use a linear AABB scan; M1+ may swap in the rtree virtual table.
    """
    return [
        n
        for n in all_nodes(conn)
        if _aabb_overlap(bbox_min, bbox_max, n.bbox_min, n.bbox_max, dilation)
    ]


def update_node(
    conn: sqlite3.Connection,
    node_id: int,
    *,
    label: str,
    labels: list[tuple[str, float]],
    confidence: float,
    centroid: Vec3,
    bbox_min: Vec3,
    bbox_max: Vec3,
    feature: np.ndarray,
    n_obs: int,
    t_last: float,
) -> None:
    conn.execute(
        """UPDATE nodes SET
            label=?, labels_json=?, confidence=?,
            centroid_x=?, centroid_y=?, centroid_z=?,
            bbox_min_x=?, bbox_min_y=?, bbox_min_z=?,
            bbox_max_x=?, bbox_max_y=?, bbox_max_z=?,
            feature=?, n_obs=?, t_last=?
           WHERE id=?""",
        (
            label,
            json.dumps(labels),
            confidence,
            centroid[0],
            centroid[1],
            centroid[2],
            bbox_min[0],
            bbox_min[1],
            bbox_min[2],
            bbox_max[0],
            bbox_max[1],
            bbox_max[2],
            _vec_to_blob(feature),
            n_obs,
            t_last,
            node_id,
        ),
    )
    _vec.upsert(conn, node_id, feature)


def stats(conn: sqlite3.Connection) -> StoreStats:
    def count(t: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) AS c FROM {t}").fetchone()["c"])

    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    return StoreStats(
        n_nodes=count("nodes"),
        n_edges=count("edges"),
        n_obs=count("observations"),
        n_episodes=count("episodes"),
        store_bytes=page_count * page_size,
    )
