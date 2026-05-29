"""sqlite-vec ANN path (V1). Skipped unless the [vec] extra is installed."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("sqlite_vec")

from spatialmem import Detection, SpatialMemory
from spatialmem import vec as _vec

DIM = 16


def _seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


class StubEncoder:
    @property
    def dim(self) -> int:
        return DIM

    def encode_text(self, texts) -> np.ndarray:
        rows = []
        for t in texts:
            rng = np.random.default_rng(_seed(t))
            v = rng.standard_normal(DIM).astype("float32")
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype="float32")


def _det(enc: StubEncoder, label: str, center) -> Detection:
    cx, cy, cz = center
    return Detection(
        label=label,
        feature=enc.encode_text([label])[0],
        center_xyz=center,
        bbox_min=(cx - 0.05, cy - 0.05, cz - 0.05),
        bbox_max=(cx + 0.05, cy + 0.05, cz + 0.05),
        confidence=0.9,
        ts=1.0,
    )


def _open(tmp_path, enc):
    return SpatialMemory.open(tmp_path / "v.smem", embedding_dim=DIM, encoder=enc)


def test_index_active_and_populated(tmp_path) -> None:
    enc = StubEncoder()
    with _open(tmp_path, enc) as mem:
        mem.add_detections([_det(enc, "mug", (0, 0, 0)), _det(enc, "kettle", (5, 0, 0))])
        mem.commit()
        assert _vec.enabled(mem._conn)
        n = mem._conn.execute("SELECT COUNT(*) FROM node_vec").fetchone()[0]
        assert n == mem.stats().n_nodes == 2


def test_ann_query_ranks(tmp_path) -> None:
    enc = StubEncoder()
    with _open(tmp_path, enc) as mem:
        mem.add_detections(
            [
                _det(enc, "mug", (0, 0, 0)),
                _det(enc, "kettle", (5, 0, 0)),
                _det(enc, "fridge", (10, 0, 0)),
            ]
        )
        mem.commit()
        res = mem.query("mug")
        assert res.nodes[0].label == "mug"
        assert res.nodes[0].score > 0.99  # exact embedding match, cosine ~1


def test_delete_removes_from_index(tmp_path) -> None:
    enc = StubEncoder()
    with _open(tmp_path, enc) as mem:
        mem.add_detections([_det(enc, "mug", (0, 0, 0))])
        mem.commit()
        node = mem.recent(n=1)[0]
        mem.forget(node.id)
        n = mem._conn.execute("SELECT COUNT(*) FROM node_vec").fetchone()[0]
        assert n == 0


def test_linear_fallback_when_index_absent(tmp_path) -> None:
    enc = StubEncoder()
    with _open(tmp_path, enc) as mem:
        mem.add_detections([_det(enc, "mug", (0, 0, 0)), _det(enc, "kettle", (5, 0, 0))])
        mem.commit()
        mem._conn.execute("DROP TABLE node_vec")  # force linear path
        assert not _vec.enabled(mem._conn)
        res = mem.query("mug")
        assert res.nodes[0].label == "mug"
