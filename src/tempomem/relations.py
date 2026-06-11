"""Spatial relation inference between object nodes — geometry only, no learning.

Derives `near` / `on` / `under` edges from node centroids and bounding boxes.
This is memory structure (scene-graph relations), not perception: it reads the
already-fused nodes and writes `edges`. Deterministic for a fixed graph.

- near:  centroids within `near_m` (symmetric → both directions stored)
- on:    A's bottom rests near B's top with x/y overlap and A above B  (A on B)
- under: the inverse of every `on` edge                                (B under A)
"""

from __future__ import annotations

import sqlite3

import numpy as np

from . import store

AUTO_TYPES = ["near", "on", "under"]


def _xy_overlap(a: store.NodeRow, b: store.NodeRow) -> bool:
    return (
        a.bbox_min[0] <= b.bbox_max[0]
        and b.bbox_min[0] <= a.bbox_max[0]
        and a.bbox_min[1] <= b.bbox_max[1]
        and b.bbox_min[1] <= a.bbox_max[1]
    )


def infer(conn: sqlite3.Connection, *, near_m: float = 0.6, on_gap_m: float = 0.08) -> int:
    """Recompute geometric relations over object nodes. Returns edges written.

    Clears prior auto edges (near/on/under) first, so it is idempotent.
    """
    store.clear_edges_by_type(conn, AUTO_TYPES)
    nodes = [n for n in store.all_nodes(conn) if n.type == "object"]
    written = 0

    def add(src: int, dst: int, type_: str, conf: float, t: float) -> None:
        nonlocal written
        store.upsert_edge(conn, src, dst, type_, conf, t)
        written += 1

    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            conf = min(a.confidence, b.confidence)
            t = max(a.t_last, b.t_last)
            dist = float(np.linalg.norm(np.asarray(a.centroid) - np.asarray(b.centroid)))
            if dist <= near_m:
                add(a.id, b.id, "near", conf, t)
                add(b.id, a.id, "near", conf, t)
            if _xy_overlap(a, b):
                if abs(a.bbox_min[2] - b.bbox_max[2]) <= on_gap_m and a.centroid[2] > b.centroid[2]:
                    add(a.id, b.id, "on", conf, t)
                    add(b.id, a.id, "under", conf, t)
                elif (
                    abs(b.bbox_min[2] - a.bbox_max[2]) <= on_gap_m and b.centroid[2] > a.centroid[2]
                ):
                    add(b.id, a.id, "on", conf, t)
                    add(a.id, b.id, "under", conf, t)
    return written
