"""Query layer: result types + retrievers + router.

M0: spatial (linear bbox/centroid scan) + temporal (recent). Semantic is M1.
See spec/QUERY-ROUTER.md.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .. import store

Vec3 = tuple[float, float, float]
Intent = Literal["semantic", "spatial", "temporal", "hybrid"]

_SPATIAL_RE = re.compile(r"\b(near|next to|on|in|under|above|inside|beside)\b|附近|旁边|上面|里面")
_TEMPORAL_RE = re.compile(r"\b(last|recent|recently|earlier|yesterday|ago)\b|刚才|最近|上次")


@dataclass
class NodeHit:
    id: int
    label: str
    center_xyz: Vec3
    confidence: float
    score: float
    t_first: float
    t_last: float


@dataclass
class QueryResult:
    nodes: list[NodeHit]
    intent_used: Intent
    debug: dict[str, Any] = field(default_factory=dict)


def _hit(n: store.NodeRow, score: float) -> NodeHit:
    return NodeHit(
        id=n.id,
        label=n.label,
        center_xyz=n.centroid,
        confidence=n.confidence,
        score=score,
        t_first=n.t_first,
        t_last=n.t_last,
    )


def detect_intent(text: str) -> Intent:
    spatial = bool(_SPATIAL_RE.search(text.lower()))
    temporal = bool(_TEMPORAL_RE.search(text.lower()))
    if spatial and temporal:
        return "hybrid"
    if spatial:
        return "spatial"
    if temporal:
        return "temporal"
    return "semantic"


def recent(conn: sqlite3.Connection, *, n: int = 10) -> list[NodeHit]:
    nodes = store.recent_nodes(conn, n)
    out: list[NodeHit] = []
    for i, nd in enumerate(nodes):
        out.append(_hit(nd, 1.0 - i / max(len(nodes), 1)))
    return out


def spatial(
    conn: sqlite3.Connection,
    *,
    near: Vec3 | None = None,
    radius: float | None = None,
    k: int = 100,
) -> list[NodeHit]:
    nodes = store.all_nodes(conn)
    if near is None:
        return [_hit(n, n.confidence) for n in nodes[:k]]
    p = np.asarray(near, dtype=np.float64)
    scored: list[tuple[float, store.NodeRow]] = []
    for n in nodes:
        d = float(np.linalg.norm(np.asarray(n.centroid) - p))
        if radius is not None and d > radius:
            continue
        scored.append((d, n))
    scored.sort(key=lambda t: (t[0], t[1].id))
    span = radius if radius else (max((d for d, _ in scored), default=1.0) or 1.0)
    return [_hit(n, max(0.0, 1.0 - d / span)) for d, n in scored[:k]]


def semantic_keyword(conn: sqlite3.Connection, text: str, *, k: int = 10) -> list[NodeHit]:
    """M0 placeholder for semantic search: case-insensitive label substring match.

    Real CLIP-text ANN over node features lands in M1.
    """
    terms = [t for t in re.split(r"\W+", text.lower()) if t]
    nodes = store.all_nodes(conn)
    scored: list[tuple[float, store.NodeRow]] = []
    for n in nodes:
        lab = n.label.lower()
        hits = sum(1 for t in terms if t in lab)
        if hits:
            scored.append((hits + n.confidence, n))
    scored.sort(key=lambda t: (-t[0], t[1].id))
    return [_hit(n, s) for s, n in scored[:k]]


def query(conn: sqlite3.Connection, text: str, *, k: int = 10, intent: str = "auto") -> QueryResult:
    used: Intent = detect_intent(text) if intent == "auto" else intent  # type: ignore[assignment]
    if used == "temporal":
        hits = recent(conn, n=k)
    elif used == "spatial":
        # No anchor coords in plain text at M0 -> fall back to keyword over labels.
        hits = semantic_keyword(conn, text, k=k) or spatial(conn, k=k)
    else:  # semantic or hybrid
        hits = semantic_keyword(conn, text, k=k)
    return QueryResult(nodes=hits, intent_used=used, debug={"text": text})
