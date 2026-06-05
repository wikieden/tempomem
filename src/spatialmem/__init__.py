"""SpatialMem — spatial memory layer for AI agents.

Public API. See spec/API.md. Detections-in ingest, incremental fusion, spatial
/ temporal / semantic query, BYO encoder + verbalizer, sqlite-vec ANN, decay /
split, the `PerceptionAdapter` seam (`add_frame`), and dataset streaming.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from . import fusion, persist, relations, serialize, store
from ._errors import (
    AdapterError,
    BadDetectionError,
    IngestError,
    QueryError,
    SchemaMismatchError,
    SpatialMemError,
    StoreError,
)
from .config import FusionConfig, SpatialMemConfig
from .datasets import DatasetSource, HashEncoder, SyntheticScene, stream
from .encoders import Encoder
from .frame import Detection, Observation
from .perception import PerceptionAdapter
from .query import NodeHit, QueryResult
from .query import detect_intent as _detect_intent
from .query import query as _query
from .query import recent as _recent
from .query import semantic_keyword as _semantic_keyword
from .query import semantic_vec as _semantic_vec
from .query import spatial as _spatial
from .store import StoreStats
from .verbalize import Verbalizer
from .verbalize import build_answer_prompt as _build_answer_prompt

__version__ = "0.1.0a1"

__all__ = [
    "AdapterError",
    "BadDetectionError",
    "CommitStats",
    "DatasetSource",
    "Detection",
    "Encoder",
    "FusionConfig",
    "HashEncoder",
    "IngestError",
    "NodeHit",
    "Observation",
    "PerceptionAdapter",
    "QueryError",
    "QueryResult",
    "SchemaMismatchError",
    "SpatialMemConfig",
    "SpatialMemError",
    "SpatialMemory",
    "StoreError",
    "StoreStats",
    "SyntheticScene",
    "Verbalizer",
    "__version__",
    "stream",
]


@dataclass
class CommitStats:
    observations_committed: int
    nodes_after: int
    elapsed_ms: float


def _now() -> float:
    return time.time()


def _node_hit(n: store.NodeRow) -> NodeHit:
    return NodeHit(
        id=n.id,
        label=n.label,
        center_xyz=n.centroid,
        confidence=n.confidence,
        score=n.confidence,
        t_first=n.t_first,
        t_last=n.t_last,
    )


def _unit_feature(dim: int) -> np.ndarray:
    """A valid L2-normalized placeholder feature (e0) for region nodes."""
    v = np.zeros(dim, dtype="float32")
    v[0] = 1.0
    return v


class SpatialMemory:
    """A persistent spatial scene-graph memory backed by a single .smem file."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_dim: int,
        readonly: bool,
        config: SpatialMemConfig | None = None,
        encoder: Encoder | None = None,
        verbalizer: Verbalizer | None = None,
        adapter: PerceptionAdapter | None = None,
    ) -> None:
        if encoder is not None and encoder.dim != embedding_dim:
            raise SchemaMismatchError(
                f"encoder dim {encoder.dim} != store embedding_dim {embedding_dim}"
            )
        self._conn = conn
        self._dim = embedding_dim
        self._readonly = readonly
        self._cfg = config or SpatialMemConfig()
        self._encoder = encoder
        self._verbalizer = verbalizer
        self._adapter = adapter
        self._pending: list[int] = []  # observation ids awaiting fusion

    # ---- lifecycle -------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | os.PathLike,
        *,
        embedding_dim: int = 512,
        create: bool = True,
        readonly: bool = False,
        config: SpatialMemConfig | None = None,
        encoder: Encoder | None = None,
        verbalizer: Verbalizer | None = None,
        adapter: PerceptionAdapter | None = None,
    ) -> SpatialMemory:
        conn = persist.connect(path, embedding_dim=embedding_dim, readonly=readonly, create=create)
        return cls(conn, embedding_dim, readonly, config, encoder, verbalizer, adapter)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> SpatialMemory:
        return self

    def __exit__(self, *_a: object) -> None:
        self.close()

    # ---- ingest ----------------------------------------------------------

    def add_detections(
        self, dets: list[Detection] | tuple[Detection, ...], *, episode: str | None = None
    ) -> list[int]:
        if self._readonly:
            raise StoreError("store opened read-only")
        session = episode or "default"
        obs_ids: list[int] = []
        for det in dets:
            if det.dim != self._dim:
                raise BadDetectionError(
                    f"detection feature dim {det.dim} != store embedding_dim {self._dim}"
                )
            ts = det.ts if det.ts is not None else _now()
            ep = store.ensure_episode(self._conn, session, ts)
            oid = store.insert_observation(
                self._conn,
                episode_id=ep,
                ts=ts,
                label=det.label,
                confidence=det.confidence,
                center=det.center_xyz,
                bbox_min=det.bbox_min,
                bbox_max=det.bbox_max,
                feature=det.feature,
                mask_rle=det.mask_rle,
                aux=det.aux,
            )
            obs_ids.append(oid)
            self._pending.append(oid)
        return obs_ids

    def add_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        *,
        intrinsics: np.ndarray | None = None,
        adapter: PerceptionAdapter | None = None,
        episode: str | None = None,
    ) -> list[int]:
        """Run a posed RGB-D frame through a PerceptionAdapter, then ingest.

        Pass `adapter=` or set one at open(). Detections still require commit().
        """
        ad = adapter or self._adapter
        if ad is None:
            raise IngestError(
                "add_frame needs a PerceptionAdapter: open(adapter=...) or add_frame(adapter=...)"
            )
        dets = ad.process_frame(rgb, depth, pose, intrinsics)
        return self.add_detections(dets, episode=episode)

    def commit(self, *, timeout_s: float = 30.0) -> CommitStats:
        t0 = _now()
        n = 0
        for oid in self._pending:
            r = self._conn.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone()
            if r is None:
                continue
            obs = Observation(
                id=int(r["id"]),
                episode_id=int(r["episode_id"]),
                ts=float(r["ts"]),
                label=r["label"],
                confidence=float(r["confidence"]),
                center_xyz=(r["center_x"], r["center_y"], r["center_z"]),
                bbox_min=(r["bbox_min_x"], r["bbox_min_y"], r["bbox_min_z"]),
                bbox_max=(r["bbox_max_x"], r["bbox_max_y"], r["bbox_max_z"]),
                feature=store._blob_to_vec(r["feature"]),
            )
            fusion.ingest_observation(self._conn, obs, self._cfg.fusion)
            n += 1
        self._pending.clear()
        self._conn.commit()
        st = store.stats(self._conn)
        return CommitStats(
            observations_committed=n,
            nodes_after=st.n_nodes,
            elapsed_ms=(_now() - t0) * 1000.0,
        )

    # ---- query -----------------------------------------------------------

    def query(self, text: str, *, k: int = 10, intent: str = "auto") -> QueryResult:
        try:
            used = _detect_intent(text) if intent == "auto" else intent
            if used in ("semantic", "hybrid") and self._encoder is not None:
                qv = self._encoder.encode_text([text])[0]
                hits = _semantic_vec(self._conn, qv, k=k)
                return QueryResult(
                    nodes=hits, intent_used=used, debug={"text": text, "encoder": True}
                )
            return _query(self._conn, text, k=k, intent=intent)
        except sqlite3.Error as e:  # pragma: no cover
            raise QueryError(str(e)) from e

    def semantic(self, text: str, *, k: int = 10) -> list[NodeHit]:
        """Semantic retrieval. Uses the encoder when one is configured (cosine
        over node features); otherwise falls back to label keyword match.
        """
        if self._encoder is not None:
            qv = self._encoder.encode_text([text])[0]
            return _semantic_vec(self._conn, qv, k=k)
        return _semantic_keyword(self._conn, text, k=k)

    def answer(self, query: str, *, k: int = 8, verbalizer: Verbalizer | None = None) -> str:
        """Retrieve relevant nodes, serialize a scene prompt, and ask a BYO LLM.

        Pass `verbalizer=` or set one at open(). Raises if none is configured.
        """
        vb = verbalizer or self._verbalizer
        if vb is None:
            raise QueryError(
                "answer() needs a verbalizer: open(verbalizer=...) or answer(verbalizer=...)"
            )
        hits = self.semantic(query, k=k)
        scene = serialize.to_prompt(self._conn)
        prompt = _build_answer_prompt(query, scene, hits)
        return vb.complete(prompt)

    def spatial(
        self,
        *,
        near: tuple[float, float, float] | None = None,
        radius: float | None = None,
        k: int = 100,
    ) -> list[NodeHit]:
        return _spatial(self._conn, near=near, radius=radius, k=k)

    def recent(self, *, n: int = 10) -> list[NodeHit]:
        return _recent(self._conn, n=n)

    def serialize(self, *, format: str = "prompt", root: int | None = None, k_hops: int = 2) -> str:
        if format == "json":
            return serialize.dump_json(self._conn, self._dim)
        if format == "prompt":
            return serialize.to_prompt(self._conn, root=root, k_hops=k_hops)
        raise QueryError(f"unknown serialize format: {format}")

    # ---- maintenance -----------------------------------------------------

    def forget(self, node_id: int) -> None:
        store.delete_node(self._conn, node_id)
        self._conn.commit()

    def decay(
        self, *, half_life_days: float = 30.0, min_conf: float = 0.1, now: float | None = None
    ) -> tuple[int, int]:
        """Age-decay node confidence; prune nodes that fall below `min_conf`.

        Returns (n_decayed, n_pruned). `now` defaults to wall-clock time.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        result = store.decay_and_prune(
            self._conn,
            now=now if now is not None else _now(),
            half_life_days=half_life_days,
            min_conf=min_conf,
        )
        self._conn.commit()
        return result

    def resplit(self) -> tuple[int, int]:
        """Scan all nodes; split any whose member observations form two
        separated clusters (config `tau_split_m` / `min_split_obs`).

        Returns (nodes_split, new_nodes_created).
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        node_ids = [n.id for n in store.all_nodes(self._conn)]
        split = 0
        created = 0
        for nid in node_ids:
            new = fusion.split_node(self._conn, nid, self._cfg.fusion)
            if new:
                split += 1
                created += len(new)
        self._conn.commit()
        return split, created

    # ---- hierarchy -------------------------------------------------------

    def define_region(
        self,
        label: str,
        bbox_min: tuple[float, float, float],
        bbox_max: tuple[float, float, float],
        *,
        type_: str = "room",
    ) -> int:
        """Create (or redefine) a region node over a bbox and adopt every object
        whose centroid falls inside it as a child (`parent_id`).

        The region's feature is the encoder embedding of `label` when an encoder
        is set (so `query("kitchen")` can find it), else the mean of its
        members' features. Returns the region node id.

        Idempotent by `(label, type_)`: redefining an existing region updates it
        in place and re-derives membership (old children are released first).
        Membership is single-parent — if regions overlap, an object's centroid
        is adopted by whichever region was defined last.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        kids = store.nodes_in_bbox(self._conn, bbox_min, bbox_max, type_="object")
        if self._encoder is not None:
            feat = self._encoder.encode_text([label])[0]
        elif kids:
            feats = [store.node_feature(self._conn, k.id) for k in kids]
            valid = [x for x in feats if x is not None and x.shape[0] == self._dim]
            if valid:
                f = np.mean(valid, axis=0)
                nrm = float(np.linalg.norm(f))
                feat = f / nrm if nrm > 0 else _unit_feature(self._dim)
            else:
                feat = _unit_feature(self._dim)
        else:
            feat = _unit_feature(self._dim)
        centroid = tuple((bbox_min[i] + bbox_max[i]) / 2 for i in range(3))
        t = max((k.t_last for k in kids), default=_now())

        existing = self._conn.execute(
            "SELECT id FROM nodes WHERE label=? AND type=? ORDER BY id LIMIT 1", (label, type_)
        ).fetchone()
        if existing is not None:
            region_id = int(existing["id"])
            for c in store.children(self._conn, region_id):
                store.set_parent(self._conn, c.id, None)  # release stale membership
            store.update_node(
                self._conn,
                region_id,
                label=label,
                labels=[(label, 1.0)],
                confidence=1.0,
                centroid=centroid,  # type: ignore[arg-type]
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                feature=feat,
                n_obs=0,
                t_last=t,
            )
        else:
            region_id = store.insert_node(
                self._conn,
                type_=type_,
                label=label,
                labels=[(label, 1.0)],
                confidence=1.0,
                centroid=centroid,  # type: ignore[arg-type]
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                feature=feat,
                n_obs=0,
                t_first=t,
                t_last=t,
            )
        for k in kids:
            store.set_parent(self._conn, k.id, region_id)
        self._conn.commit()
        return region_id

    def contents(self, region: int | str) -> list[NodeHit]:
        """Children of a region — by node id or by region label."""
        if isinstance(region, str):
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE label=? AND type!='object' ORDER BY id LIMIT 1",
                (region,),
            ).fetchone()
            if row is None:
                return []
            rid = int(row["id"])
        else:
            rid = region
        return [_node_hit(n) for n in store.children(self._conn, rid)]

    # ---- relations -------------------------------------------------------

    def relate(self, *, near_m: float = 0.6, on_gap_m: float = 0.08) -> int:
        """Infer geometric relations (near / on / under) over object nodes and
        store them as edges. Idempotent; returns the number of edges written.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        n = relations.infer(self._conn, near_m=near_m, on_gap_m=on_gap_m)
        self._conn.commit()
        return n

    def related(self, node: int | str, *, rel: str | None = None) -> list[tuple[NodeHit, str]]:
        """Neighbors of a node via relation edges, as (neighbor, relation_type).

        `node` is a node id or an object label. `rel` filters to one relation
        type (e.g. "on", "near", "under").
        """
        if isinstance(node, str):
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE label=? AND type='object' ORDER BY id LIMIT 1",
                (node,),
            ).fetchone()
            if row is None:
                return []
            nid = int(row["id"])
        else:
            nid = node
        out: list[tuple[NodeHit, str]] = []
        for dst, type_, _conf in store.edges_from(self._conn, nid, rel):
            n = store.get_node(self._conn, dst)
            if n is not None:
                out.append((_node_hit(n), type_))
        return out

    def stats(self) -> StoreStats:
        return store.stats(self._conn)
