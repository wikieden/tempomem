"""Fusion arbiter — M0 STUB.

M0 policy: every observation becomes a new node. No merge/dedup yet.
Real scoring (geom/iou/sem/label) + merge transaction land in M1.
See spec/FUSION-ARBITER.md.
"""

from __future__ import annotations

import sqlite3

from . import store
from .frame import Observation


def ingest_observation(conn: sqlite3.Connection, obs: Observation) -> int:
    """STUB: create one node per observation. Returns node id."""
    node_id = store.insert_node(
        conn,
        type_="object",
        label=obs.label,
        labels=[(obs.label, 1.0)],
        confidence=obs.confidence,
        centroid=obs.center_xyz,
        bbox_min=obs.bbox_min,
        bbox_max=obs.bbox_max,
        feature=obs.feature,
        n_obs=1,
        t_first=obs.ts,
        t_last=obs.ts,
    )
    store.link_node_obs(conn, node_id, obs.id, obs.ts)
    return node_id
