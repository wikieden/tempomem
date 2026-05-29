"""SpatialMem — spatial memory layer for AI agents.

Public API. See spec/API.md. M0 = detections-in only; perception adapters
(M2), real fusion (M1), and semantic ANN (M1) are not wired yet.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass

from . import fusion, persist, serialize, store
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
from .frame import Detection, Observation
from .query import NodeHit, QueryResult
from .query import query as _query
from .query import recent as _recent
from .query import spatial as _spatial
from .store import StoreStats

__version__ = "0.1.0a1"

__all__ = [
    "AdapterError",
    "BadDetectionError",
    "CommitStats",
    "Detection",
    "FusionConfig",
    "IngestError",
    "NodeHit",
    "Observation",
    "QueryError",
    "QueryResult",
    "SchemaMismatchError",
    "SpatialMemConfig",
    "SpatialMemError",
    "SpatialMemory",
    "StoreError",
    "StoreStats",
    "__version__",
]


@dataclass
class CommitStats:
    observations_committed: int
    nodes_after: int
    elapsed_ms: float


def _now() -> float:
    return time.time()


class SpatialMemory:
    """A persistent spatial scene-graph memory backed by a single .smem file."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_dim: int,
        readonly: bool,
        config: SpatialMemConfig | None = None,
    ) -> None:
        self._conn = conn
        self._dim = embedding_dim
        self._readonly = readonly
        self._cfg = config or SpatialMemConfig()
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
    ) -> SpatialMemory:
        conn = persist.connect(path, embedding_dim=embedding_dim, readonly=readonly, create=create)
        return cls(conn, embedding_dim, readonly, config)

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
            return _query(self._conn, text, k=k, intent=intent)
        except sqlite3.Error as e:  # pragma: no cover
            raise QueryError(str(e)) from e

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

    def stats(self) -> StoreStats:
        return store.stats(self._conn)
