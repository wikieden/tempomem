"""Chronotope — spatial memory layer for AI agents.

Public API. See spec/API.md. Detections-in ingest, incremental fusion, spatial
/ temporal / semantic query, BYO encoder + verbalizer, sqlite-vec ANN, decay /
split, the `PerceptionAdapter` seam (`add_frame`), and dataset streaming.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from . import fusion, persist, relations, serialize, store
from ._errors import (
    AdapterError,
    BadDetectionError,
    ChronotopeError,
    IngestError,
    QueryError,
    SchemaMismatchError,
    StoreError,
    ToolError,
)
from .config import ChronotopeConfig, FusionConfig
from .datasets import DatasetSource, HashEncoder, SyntheticScene, stream
from .encoders import Encoder
from .frame import Detection, Observation
from .perception import PerceptionAdapter
from .query import NodeHit, QueryResult
from .query import detect_intent as _detect_intent
from .query import query as _query
from .query import recent as _recent
from .query import relational as _relational
from .query import semantic_keyword as _semantic_keyword
from .query import semantic_vec as _semantic_vec
from .query import spatial as _spatial
from .store import StoreStats
from .tools import ChronotopeTools
from .verbalize import Verbalizer
from .verbalize import build_answer_prompt as _build_answer_prompt

__version__ = "0.1.0a1"

_log = logging.getLogger("tempomem.memory")

__all__ = [
    "AdapterError",
    "BadDetectionError",
    "ChangeSet",
    "ChronotopeConfig",
    "ChronotopeError",
    "ChronotopeTools",
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
    "StoreError",
    "StoreStats",
    "SyntheticScene",
    "TempoMem",
    "ToolError",
    "Verbalizer",
    "__version__",
    "stream",
]


@dataclass
class CommitStats:
    observations_committed: int
    nodes_after: int
    elapsed_ms: float


@dataclass
class ChangeSet:
    """What changed in the memory since a timestamp."""

    new: list[NodeHit]
    seen_again: list[NodeHit]


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


class TempoMem:
    """A persistent spatial scene-graph memory backed by a single .smem file."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_dim: int,
        readonly: bool,
        config: ChronotopeConfig | None = None,
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
        self._cfg = config or ChronotopeConfig()
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
        config: ChronotopeConfig | None = None,
        encoder: Encoder | None = None,
        verbalizer: Verbalizer | None = None,
        adapter: PerceptionAdapter | None = None,
    ) -> TempoMem:
        conn = persist.connect(path, embedding_dim=embedding_dim, readonly=readonly, create=create)
        return cls(conn, embedding_dim, readonly, config, encoder, verbalizer, adapter)

    def close(self) -> None:
        if self._conn is not None:
            self._flush_pending()  # fuse staged obs before the final commit
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> TempoMem:
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
        # spec: ENGINEERING.md §auto-flush — if max_pending_obs is set and the
        # threshold is reached, fuse immediately so _pending never grows without
        # bound in a long-running process that forgets to call commit().
        limit = self._cfg.max_pending_obs
        if limit is not None and len(self._pending) >= limit:
            _log.warning(
                "tempomem: auto-flushing %d pending observations "
                "(max_pending_obs=%d); call commit() explicitly to avoid this",
                len(self._pending),
                limit,
            )
            self.commit()
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

    def _flush_pending(self) -> None:
        """Fuse + persist any staged observations before a maintenance commit.

        Fuse-before-persist invariant: `add_detections()`/`add_frame()` stage
        observation rows in `self._pending` that are not yet linked to a node
        (fusion is deferred to `commit()`). Every maintenance/structural mutator
        below issues its own `conn.commit()`; because the sqlite connection runs
        in implicit-transaction mode, that commit would flush the staged rows to
        disk *unfused* — leaving orphan observations on a crash and showing a
        half-ingested store to whatever ran in between. Draining `_pending`
        through `commit()` first guarantees the store is never persisted with an
        observation that is not linked to a node, and that maintenance always
        runs against a fully fused store. `commit()` is the sole drainer of
        `_pending`; no mutating call is valid while observations are pending.
        """
        if self._pending:
            self.commit()

    # ---- query -----------------------------------------------------------

    def query(self, text: str, *, k: int = 10, intent: str = "auto") -> QueryResult:
        try:
            if intent in ("auto", "spatial", "relational"):
                rel = _relational(self._conn, text, k=k)
                if rel is not None:
                    return rel  # relation phrase + anchor object matched
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

    def serialize(
        self,
        *,
        format: str = "prompt",
        root: int | None = None,
        k_hops: int = 2,
        relations: bool = True,
        max_tokens: int | None = None,
        node_ids: set[int] | None = None,
    ) -> str:
        if format == "json":
            return serialize.dump_json(self._conn, self._dim)
        if format == "prompt":
            return serialize.to_prompt(
                self._conn,
                root=root,
                k_hops=k_hops,
                relations=relations,
                max_tokens=max_tokens,
                node_ids=node_ids,
            )
        raise QueryError(f"unknown serialize format: {format}")

    # ---- maintenance -----------------------------------------------------

    def forget(self, node_id: int) -> None:
        self._flush_pending()
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
        self._flush_pending()
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
        self._flush_pending()
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

    def consolidate(self) -> int:
        """Merge near-duplicate object nodes that fusion missed (e.g. created in
        separate sessions or just under threshold). Returns the number of merges.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        self._flush_pending()
        n = fusion.consolidate(self._conn, self._cfg.fusion)
        self._conn.commit()
        return n

    def salient(self, *, n: int = 10) -> list[NodeHit]:
        """Top-n nodes by salience = recency * confidence * evidence (n_obs).
        Use to prioritize what matters in a crowded memory.
        """
        nodes = store.all_nodes(self._conn)
        if not nodes:
            return []
        ts = [x.t_last for x in nodes]
        lo, hi = min(ts), max(ts)
        span = (hi - lo) or 1.0
        scored = []
        for nd in nodes:
            rec = (nd.t_last - lo) / span
            sal = (0.5 + 0.5 * rec) * nd.confidence * (1.0 + 0.1 * nd.n_obs)
            scored.append((sal, nd))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [
            NodeHit(
                id=nd.id,
                label=nd.label,
                center_xyz=nd.centroid,
                confidence=nd.confidence,
                score=float(sal),
                t_first=nd.t_first,
                t_last=nd.t_last,
            )
            for sal, nd in scored[:n]
        ]

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
        self._flush_pending()
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
        self._flush_pending()
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

    # ---- update / history ------------------------------------------------

    def update(
        self,
        node_id: int,
        *,
        label: str | None = None,
        center_xyz: tuple[float, float, float] | None = None,
        confidence: float | None = None,
    ) -> None:
        """Correct a node in place (Mem0-style update). Only the given fields
        change; moving `center_xyz` shifts the bbox by the same delta, keeping
        its extent. Raises StoreError if the node does not exist.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        self._flush_pending()
        n = store.get_node(self._conn, node_id)
        if n is None:
            raise StoreError(f"node {node_id} not found")
        new_label = n.label if label is None else label
        new_labels = n.labels if label is None else [(label, 1.0)]
        new_conf = n.confidence if confidence is None else confidence
        if new_conf < 0.0 or new_conf > 1.0:
            raise StoreError(f"confidence {new_conf} not in [0, 1]")
        if center_xyz is not None:
            half = tuple((n.bbox_max[i] - n.bbox_min[i]) / 2 for i in range(3))
            centroid = center_xyz
            bbox_min = tuple(center_xyz[i] - half[i] for i in range(3))
            bbox_max = tuple(center_xyz[i] + half[i] for i in range(3))
        else:
            centroid, bbox_min, bbox_max = n.centroid, n.bbox_min, n.bbox_max
        feat = store.node_feature(self._conn, node_id)
        assert feat is not None
        store.update_node(
            self._conn,
            node_id,
            label=new_label,
            labels=new_labels,
            confidence=new_conf,
            centroid=centroid,  # type: ignore[arg-type]
            bbox_min=bbox_min,  # type: ignore[arg-type]
            bbox_max=bbox_max,  # type: ignore[arg-type]
            feature=feat,
            n_obs=n.n_obs,
            t_last=n.t_last,
        )
        self._conn.commit()

    def merge(self, other: str | os.PathLike, *, episode: str | None = None) -> CommitStats:
        """Merge another `.smem` store's objects into this one — re-entering a
        space continues the same memory instead of starting over. Each object
        node of `other` is fed through fusion as one observation, so the same
        physical object seen in both sessions converges to a single node; new
        objects are added. Regions are not merged. Returns the commit stats.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        src = TempoMem.open(other, embedding_dim=self._dim, create=False, readonly=True)
        try:
            dets: list[Detection] = []
            for n in store.all_nodes(src._conn):
                if n.type != "object":
                    continue
                f = store.node_feature(src._conn, n.id)
                if f is None:
                    continue
                dets.append(
                    Detection(
                        label=n.label,
                        feature=f,
                        center_xyz=n.centroid,
                        bbox_min=n.bbox_min,
                        bbox_max=n.bbox_max,
                        confidence=n.confidence,
                        ts=n.t_last,
                    )
                )
            self.add_detections(dets, episode=episode or "merged")
            return self.commit()
        finally:
            src.close()

    def history(self, node_id: int) -> list[Observation]:
        """The time-ordered observation trail behind a node — every sighting
        that fused into it, with its timestamp and position. Answers "where was
        it over time" / "when was it last seen" (`history(id)[-1]`).
        """
        return store.observations_for_node(self._conn, node_id)

    def last_seen(
        self, name: str, *, now: float | None = None
    ) -> tuple[str | None, float, float] | None:
        """Where was `name` last seen — ``(region_label, age_s, confidence)``.

        Exact-label lookup over object nodes (region nodes excluded); the most
        recently observed match wins. ``region_label`` is the label of the
        node's parent region, or None for a top-level node. ``age_s`` is
        seconds since the node's last fusing observation (wall clock, or the
        injected ``now``). ``confidence`` is the node's decayed confidence in
        [0, 1] — the find-family hedging criterion (< 0.7 ⇒ hedge, not assert).
        Returns None when no object with that label exists (expected absence,
        not an error).
        """
        t = _now() if now is None else now
        best: store.NodeRow | None = None
        for n in store.all_nodes(self._conn):
            # regions carry their own type ("room", ...); only detection-born
            # object nodes are find-able things
            if n.type != "object" or n.label != name:
                continue
            if best is None or n.t_last > best.t_last:
                best = n
        if best is None:
            return None
        region: str | None = None
        if best.parent_id is not None:
            parent = store.get_node(self._conn, best.parent_id)
            region = parent.label if parent is not None else None
        return region, max(0.0, t - best.t_last), best.confidence

    # ---- change detection ------------------------------------------------

    def moved(self, node_id: int) -> float:
        """Displacement (meters) of a node between its first and last
        observation — how far the object travelled across sightings. 0 for a
        node with fewer than two observations.
        """
        trail = store.observations_for_node(self._conn, node_id)
        if len(trail) < 2:
            return 0.0
        a = np.asarray(trail[0].center_xyz)
        b = np.asarray(trail[-1].center_xyz)
        return float(np.linalg.norm(b - a))

    def changes(self, since_ts: float) -> ChangeSet:
        """Nodes new or re-observed at/after `since_ts`. `new` first appeared
        then; `seen_again` existed before but was observed again since.
        """
        new: list[NodeHit] = []
        seen_again: list[NodeHit] = []
        for n in store.all_nodes(self._conn):
            if n.t_first >= since_ts:
                new.append(_node_hit(n))
            elif n.t_last >= since_ts:
                seen_again.append(_node_hit(n))
        return ChangeSet(new=new, seen_again=seen_again)

    def stale(self, before_ts: float) -> list[NodeHit]:
        """Nodes not observed since `before_ts` — candidates for "gone"."""
        return [_node_hit(n) for n in store.all_nodes(self._conn) if n.t_last < before_ts]

    def stats(self) -> StoreStats:
        return store.stats(self._conn)

    # ---- semantic edges --------------------------------------------------

    def _resolve_nid(self, label: str) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM nodes WHERE label=? ORDER BY id LIMIT 1", (label,)
        ).fetchone()
        return int(row["id"]) if row else None

    def add_edge(
        self, src: int | str, rel: str, dst: int | str, *, ts: float | None = None
    ) -> None:
        """Write a typed directed edge src --rel--> dst in the semantic graph.

        Idempotent on (src, rel, dst); call again to update ts. Both ends can
        be a node id (int) or a node label (str, first match by label).
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        self._flush_pending()
        src_id = self._resolve_nid(src) if isinstance(src, str) else src
        dst_id = self._resolve_nid(dst) if isinstance(dst, str) else dst
        if src_id is None:
            raise StoreError(f"node not found: {src!r}")
        if dst_id is None:
            raise StoreError(f"node not found: {dst!r}")
        store.sem_edge_upsert(self._conn, src_id, rel, dst_id, ts)
        self._conn.commit()

    def get_edges(
        self,
        node: int | str,
        *,
        rel: str | None = None,
        direction: str = "out",
    ) -> list[tuple[NodeHit, str]]:
        """Typed semantic edges of a node, as (neighbor, rel_type) pairs.

        `direction="out"` (default): edges where node is source.
        `direction="in"`: edges where node is destination.
        `rel` filters to one relation type when given.
        """
        nid = self._resolve_nid(node) if isinstance(node, str) else node
        if nid is None:
            return []
        if direction == "in":
            pairs = store.sem_edges_to(self._conn, nid, rel)
        else:
            pairs = store.sem_edges_from(self._conn, nid, rel)
        out: list[tuple[NodeHit, str]] = []
        for other_id, r in pairs:
            n = store.get_node(self._conn, other_id)
            if n is not None:
                out.append((_node_hit(n), r))
        return out

    # ---- node properties -------------------------------------------------

    def set_property(
        self, node: int | str, key: str, value: object, *, ts: float | None = None
    ) -> None:
        """Write a property on a node. Overwrites previous value for the same key.

        `value` is any JSON-serialisable Python object.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        self._flush_pending()
        nid = self._resolve_nid(node) if isinstance(node, str) else node
        if nid is None:
            raise StoreError(f"node not found: {node!r}")
        store.prop_set(self._conn, nid, key, json.dumps(value), ts)
        self._conn.commit()

    def get_property(self, node: int | str, key: str) -> object | None:
        """Read current value of a property, or None if unset."""
        nid = self._resolve_nid(node) if isinstance(node, str) else node
        if nid is None:
            return None
        raw = store.prop_get(self._conn, nid, key)
        return json.loads(raw) if raw is not None else None

    # ---- semantic event timeline -----------------------------------------

    def add_event(
        self,
        type: str,
        *,
        location: int | str | None = None,
        ts: float,
        payload: dict | None = None,
    ) -> int:
        """Append a semantic event to the timeline. Returns the event id.

        `location` pins the event to a node (id or label). `payload` is any
        JSON-serialisable dict carrying event-specific data.
        """
        if self._readonly:
            raise StoreError("store opened read-only")
        self._flush_pending()
        loc_id: int | None = None
        if location is not None:
            loc_id = self._resolve_nid(location) if isinstance(location, str) else location
            if loc_id is None:
                raise StoreError(f"location node not found: {location!r}")
        eid = store.event_insert(
            self._conn, type, loc_id, ts, json.dumps(payload) if payload else None
        )
        self._conn.commit()
        return eid

    def query_events(
        self,
        type: str,
        *,
        region: int | str | None = None,
        since_ts: float | None = None,
    ) -> list[dict]:
        """Return semantic events of a given type, ordered by ts ascending.

        `region` filters to events whose location is that node.
        `since_ts` excludes events before that timestamp.
        """
        loc_id: int | None = None
        if region is not None:
            loc_id = self._resolve_nid(region) if isinstance(region, str) else region
            if loc_id is None:
                return []
        rows = store.events_query(self._conn, type, loc_id, since_ts)
        return [
            {
                "id": int(r["id"]),
                "type": r["type"],
                "location": int(r["location"]) if r["location"] is not None else None,
                "ts": float(r["ts"]),
                "payload": json.loads(r["payload"]) if r["payload"] else None,
            }
            for r in rows
        ]

    def last_changed(self, region: int | str) -> tuple[float | None, str | None]:
        """When was this region last modified? Returns (timestamp, change_type).

        Checks both direct node updates (children of the region) and smem_events
        pinned to the location. Returns (None, None) if no activity found.
        """
        rid = self._resolve_nid(region) if isinstance(region, str) else region
        if rid is None:
            return None, None
        row = self._conn.execute(
            "SELECT MAX(t_last) AS t FROM nodes WHERE parent_id=?", (rid,)
        ).fetchone()
        node_t: float | None = float(row["t"]) if row and row["t"] is not None else None

        row2 = self._conn.execute(
            "SELECT ts, type FROM smem_events WHERE location=? ORDER BY ts DESC LIMIT 1",
            (rid,),
        ).fetchone()
        event_t: float | None = float(row2["ts"]) if row2 else None
        event_type: str | None = row2["type"] if row2 else None

        if node_t is None and event_t is None:
            return None, None
        if node_t is not None and (event_t is None or node_t >= event_t):
            return node_t, "node_update"
        return event_t, event_type
