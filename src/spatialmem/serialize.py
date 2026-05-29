"""Graph serialization: JSON dump + prompt text. See spec/SCHEMA.md."""

from __future__ import annotations

import json
import sqlite3

from . import store
from .persist import SCHEMA_VERSION


def to_json(conn: sqlite3.Connection, embedding_dim: int) -> dict:
    episodes = [
        {
            "id": int(r["id"]),
            "session": r["session"],
            "start_ts": r["start_ts"],
            "end_ts": r["end_ts"],
        }
        for r in conn.execute("SELECT * FROM episodes ORDER BY id")
    ]
    nodes = [
        {
            "id": n.id,
            "type": n.type,
            "label": n.label,
            "labels": n.labels,
            "confidence": round(n.confidence, 4),
            "centroid": [round(c, 4) for c in n.centroid],
            "bbox": [list(n.bbox_min), list(n.bbox_max)],
            "n_obs": n.n_obs,
            "t_first": n.t_first,
            "t_last": n.t_last,
            "parent_id": n.parent_id,
        }
        for n in store.all_nodes(conn)
    ]
    edges = [
        {
            "src": int(r["src"]),
            "dst": int(r["dst"]),
            "type": r["type"],
            "confidence": round(float(r["confidence"]), 4),
            "t_last": float(r["t_last"]),
        }
        for r in conn.execute("SELECT * FROM edges ORDER BY id")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_dim": embedding_dim,
        "episodes": episodes,
        "nodes": nodes,
        "edges": edges,
    }


def to_prompt(conn: sqlite3.Connection, *, root: int | None = None, k_hops: int = 1) -> str:
    """Token-efficient indented text. M0: flat list (hierarchy inference is M2)."""
    nodes = store.all_nodes(conn)
    if root is not None:
        nodes = [n for n in nodes if n.id == root] or nodes
    t_now = max((n.t_last for n in nodes), default=0.0)
    lines = [f"SCENE (root={root}, k_hops={k_hops}, ts={t_now:.1f})"]
    for n in sorted(nodes, key=lambda x: (-x.t_last, x.id)):
        c = n.centroid
        lines.append(
            f'  {n.type}#{n.id} "{n.label}"  '
            f"@[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]  "
            f"t_last={n.t_last:.1f}  conf={n.confidence:.2f}"
        )
    return "\n".join(lines)


def dump_json(conn: sqlite3.Connection, embedding_dim: int, indent: int = 2) -> str:
    return json.dumps(to_json(conn, embedding_dim), indent=indent)
