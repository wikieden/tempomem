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


def _fmt_node(n: store.NodeRow, indent: int) -> str:
    c = n.centroid
    pad = "  " * indent
    return (
        f'{pad}{n.type}#{n.id} "{n.label}"  '
        f"@[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]  "
        f"t_last={n.t_last:.1f}  conf={n.confidence:.2f}"
    )


def to_prompt(
    conn: sqlite3.Connection,
    *,
    root: int | None = None,
    k_hops: int = 2,
    relations: bool = True,
    max_tokens: int | None = None,
) -> str:
    """Token-efficient indented text, grouped by hierarchy: region/room nodes
    list their child objects indented beneath them; ungrouped objects sit at the
    top level. `root` restricts the tree to one node's subtree. When `relations`
    and edges exist, each node line gets a `| <rel> <label>#<id>, …` suffix
    (e.g. `| on table#3, near kettle#2`) so an LLM sees the scene graph.

    `max_tokens` caps the output: nodes are emitted most-recent-first and the
    rest dropped with an explicit `… (N more omitted)` marker (never silent).
    """
    nodes = store.all_nodes(conn)
    by_id = {n.id: n for n in nodes}
    children: dict[int | None, list[store.NodeRow]] = {}
    for n in nodes:
        children.setdefault(n.parent_id, []).append(n)
    t_now = max((n.t_last for n in nodes), default=0.0)
    lines = [f"SCENE (root={root}, ts={t_now:.1f})"]

    def emit(n: store.NodeRow, depth: int) -> None:
        line = _fmt_node(n, depth)
        if relations:
            rels = ", ".join(
                f"{t} {by_id[d].label}#{d}"
                for d, t, _ in store.edges_from(conn, n.id)
                if d in by_id
            )
            if rels:
                line += f"  | {rels}"
        lines.append(line)
        for kid in sorted(children.get(n.id, []), key=lambda x: (-x.t_last, x.id)):
            emit(kid, depth + 1)

    if root is not None and root in by_id:
        emit(by_id[root], 1)
    else:
        tops = sorted(children.get(None, []), key=lambda x: (-x.t_last, x.id))
        for n in tops:
            emit(n, 1)

    if max_tokens is None:
        return "\n".join(lines)

    def _est(s: str) -> int:
        return max(1, len(s) // 4)  # ~4 chars/token

    kept = [lines[0]]
    used = _est(lines[0])
    dropped = 0
    for ln in lines[1:]:
        cost = _est(ln)
        if used + cost <= max_tokens:
            kept.append(ln)
            used += cost
        else:
            dropped += 1
    if dropped:
        kept.append(f"  … ({dropped} more omitted)")
    return "\n".join(kept)


def dump_json(conn: sqlite3.Connection, embedding_dim: int, indent: int = 2) -> str:
    return json.dumps(to_json(conn, embedding_dim), indent=indent)
