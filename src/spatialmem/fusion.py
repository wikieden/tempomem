"""Fusion arbiter — M1.

For each incoming observation, decide merge into an existing node, create a
new node, or reject. Deterministic given a fixed config + observation stream.
See spec/FUSION-ARBITER.md.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from . import store
from .config import FusionConfig
from .frame import Observation

Vec3 = tuple[float, float, float]


def iou3d(amin: Vec3, amax: Vec3, bmin: Vec3, bmax: Vec3) -> float:
    inter = 1.0
    for i in range(3):
        lo = max(amin[i], bmin[i])
        hi = min(amax[i], bmax[i])
        d = hi - lo
        if d <= 0:
            return 0.0
        inter *= d
    va = 1.0
    vb = 1.0
    for i in range(3):
        va *= max(amax[i] - amin[i], 0.0)
        vb *= max(bmax[i] - bmin[i], 0.0)
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


def label_compat(obs_label: str, node_labels: list[tuple[str, float]]) -> float:
    """M1 lexical label compatibility. CLIP-text scoring is M2."""
    ol = obs_label.lower()
    best = 0.0
    for lab, weight in node_labels:
        ll = lab.lower()
        if ll == ol:
            best = max(best, max(weight, 0.8))
        elif ol in ll or ll in ol:
            best = max(best, 0.5)
    return best


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def score(
    obs: Observation,
    node: store.NodeRow,
    node_feature: np.ndarray,
    cfg: FusionConfig,
) -> float:
    dist = float(np.linalg.norm(np.asarray(obs.center_xyz) - np.asarray(node.centroid)))
    s_geom = _clip01(1.0 - dist / cfg.dist_norm_m)
    s_iou = _clip01(iou3d(obs.bbox_min, obs.bbox_max, node.bbox_min, node.bbox_max))
    s_sem = _clip01(float(np.dot(obs.feature, node_feature)))
    s_label = _clip01(label_compat(obs.label, node.labels))
    return cfg.w_geom * s_geom + cfg.w_iou * s_iou + cfg.w_sem * s_sem + cfg.w_label * s_label


def _merge(
    conn: sqlite3.Connection, node: store.NodeRow, obs: Observation, cfg: FusionConfig
) -> None:
    a = cfg.centroid_alpha
    old_c = np.asarray(node.centroid, dtype=np.float64)
    new_c = old_c + a * (np.asarray(obs.center_xyz) - old_c)

    old_f = store.node_feature(conn, node.id)
    assert old_f is not None
    new_f = old_f + a * (obs.feature - old_f)
    n = float(np.linalg.norm(new_f))
    if n > 0:
        new_f = new_f / n

    bbox_min = tuple(min(node.bbox_min[i], obs.bbox_min[i]) for i in range(3))
    bbox_max = tuple(max(node.bbox_max[i], obs.bbox_max[i]) for i in range(3))

    labels = dict(node.labels)
    labels[obs.label] = labels.get(obs.label, 0.0) + obs.confidence
    tot = sum(labels.values()) or 1.0
    labels_norm = sorted(((k, v / tot) for k, v in labels.items()), key=lambda kv: (-kv[1], kv[0]))
    canonical = labels_norm[0][0]

    conf = node.confidence + (1.0 - node.confidence) * obs.confidence * cfg.conf_gain

    store.update_node(
        conn,
        node.id,
        label=canonical,
        labels=labels_norm,
        confidence=min(1.0, conf),
        centroid=(float(new_c[0]), float(new_c[1]), float(new_c[2])),
        bbox_min=bbox_min,  # type: ignore[arg-type]
        bbox_max=bbox_max,  # type: ignore[arg-type]
        feature=new_f,
        n_obs=node.n_obs + 1,
        t_last=max(node.t_last, obs.ts),
    )
    store.link_node_obs(conn, node.id, obs.id, obs.ts)


def _new_node(conn: sqlite3.Connection, obs: Observation) -> int:
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


def ingest_observation(
    conn: sqlite3.Connection, obs: Observation, cfg: FusionConfig | None = None
) -> int | None:
    """Merge into the best-matching node, create a new node, or reject.

    Returns the affected node id, or None if the observation was rejected.
    """
    cfg = cfg or FusionConfig()
    candidates = store.candidates_near(conn, obs.bbox_min, obs.bbox_max, cfg.search_dilation_m)

    best_node: store.NodeRow | None = None
    best_score = -1.0
    for node in candidates:
        nf = store.node_feature(conn, node.id)
        if nf is None:
            continue
        s = score(obs, node, nf, cfg)
        if s > best_score or (s == best_score and best_node is not None and node.id < best_node.id):
            best_score = s
            best_node = node

    if best_node is not None and best_score >= cfg.tau_merge:
        _merge(conn, best_node, obs, cfg)
        return best_node.id
    if obs.confidence < cfg.tau_obs:
        return None
    return _new_node(conn, obs)
