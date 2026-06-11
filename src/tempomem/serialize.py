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
    node_ids: set[int] | None = None,
) -> str:
    """Token-efficient indented text, grouped by hierarchy: region/room nodes
    list their child objects indented beneath them; ungrouped objects sit at the
    top level. `root` restricts the tree to one node's subtree. When `relations`
    and edges exist, each node line gets a `| <rel> <label>#<id>, …` suffix
    (e.g. `| on table#3, near kettle#2`) so an LLM sees the scene graph.

    `max_tokens` caps the output: nodes are emitted most-recent-first and the
    rest dropped with an explicit `… (N more omitted)` marker (never silent).
    `node_ids` restricts the scene to those nodes, their 1-hop relation
    neighbours, and the hierarchy ancestors of all — a query-relevant subgraph
    (keeping relational anchors like the "table" in "on the table") rather than
    the whole store.
    """
    nodes = store.all_nodes(conn)
    by_id = {n.id: n for n in nodes}
    if node_ids is not None:
        # Focused retrieval context: keep the queried nodes, their 1-hop relation
        # neighbours (so a relational hit like "what's on the table" keeps the
        # table anchor + the `on` edge), and the hierarchy ancestors of all — a
        # query-relevant subgraph, not the whole scene (VISION §2.3, OQ-6).
        seeds = set(node_ids)
        for nid in node_ids:
            for d, _t, _w in store.edges_from(conn, nid):
                if d in by_id:
                    seeds.add(d)
        keep: set[int] = set()
        for nid in seeds:
            cur = by_id.get(nid)
            while cur is not None and cur.id not in keep:
                keep.add(cur.id)
                cur = by_id.get(cur.parent_id) if cur.parent_id is not None else None
        nodes = [n for n in nodes if n.id in keep]
        by_id = {n.id: n for n in nodes}
    children: dict[int | None, list[store.NodeRow]] = {}
    for n in nodes:
        children.setdefault(n.parent_id, []).append(n)
    t_now = max((n.t_last for n in nodes), default=0.0)
    effective_root = root if (root is not None and root in by_id) else None
    header = f"SCENE (root={effective_root}, ts={t_now:.1f})"

    def emit(n: store.NodeRow, depth: int, out: list[str], seen: set[int]) -> None:
        if n.id in seen:  # defensive: a malformed parent_id cycle degrades gracefully
            return
        seen.add(n.id)
        line = _fmt_node(n, depth)
        if relations:
            rels = ", ".join(
                f"{t} {by_id[d].label}#{d}"
                for d, t, _ in store.edges_from(conn, n.id)
                if d in by_id
            )
            if rels:
                line += f"  | {rels}"
        out.append(line)
        for kid in sorted(children.get(n.id, []), key=lambda x: (-x.t_last, x.id)):
            emit(kid, depth + 1, out, seen)

    # One segment per top-level subtree (a region with its children, or a lone
    # object), most-recent-first. Budgeting keeps/drops whole subtrees so a
    # region never appears without its contents.
    if effective_root is not None:
        tops = [by_id[effective_root]]
    else:
        tops = sorted(children.get(None, []), key=lambda x: (-x.t_last, x.id))
    segments: list[list[str]] = []
    seen: set[int] = set()
    for n in tops:
        seg: list[str] = []
        emit(n, 1, seg, seen)
        segments.append(seg)

    if max_tokens is None:
        out = [header]
        for seg in segments:
            out.extend(seg)
        return "\n".join(out)

    def _est(s: str) -> int:
        return max(1, len(s) // 4)  # ~4 chars/token

    reserve = 12  # leave room for the omission marker so it always fits
    kept = [header]
    used = _est(header)
    dropped = 0
    for seg in segments:
        cost = sum(_est(line) for line in seg)
        if used + cost <= max_tokens - reserve:
            kept.extend(seg)
            used += cost
        else:
            dropped += len(seg)
    if dropped:
        kept.append(f"  … ({dropped} more omitted)")
    return "\n".join(kept)


def dump_json(conn: sqlite3.Connection, embedding_dim: int, indent: int = 2) -> str:
    return json.dumps(to_json(conn, embedding_dim), indent=indent)
