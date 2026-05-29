from __future__ import annotations

import hashlib

import numpy as np
import pytest

from spatialmem import Detection, SchemaMismatchError, SpatialMemory

DIM = 16


def _seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


class StubEncoder:
    """Deterministic text encoder: string -> seeded unit vector.

    Stands in for a real CLIP encoder so semantic retrieval is testable
    without Torch. Same string always yields the same vector.
    """

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


def test_semantic_query_ranks_by_embedding(tmp_path) -> None:
    enc = StubEncoder()
    with SpatialMemory.open(tmp_path / "s.smem", embedding_dim=DIM, encoder=enc) as mem:
        mem.add_detections(
            [
                _det(enc, "mug", (0, 0, 0)),
                _det(enc, "kettle", (5, 0, 0)),
                _det(enc, "fridge", (10, 0, 0)),
            ]
        )
        mem.commit()
        res = mem.query("mug")
        assert res.debug.get("encoder") is True
        assert res.nodes[0].label == "mug"
        assert res.nodes[0].score > 0.99  # exact embedding match


def test_semantic_method_uses_encoder(tmp_path) -> None:
    enc = StubEncoder()
    with SpatialMemory.open(tmp_path / "s2.smem", embedding_dim=DIM, encoder=enc) as mem:
        mem.add_detections([_det(enc, "kettle", (0, 0, 0)), _det(enc, "mug", (5, 0, 0))])
        mem.commit()
        hits = mem.semantic("kettle")
        assert hits[0].label == "kettle"


def test_semantic_without_encoder_falls_back_to_keyword(tmp_path) -> None:
    enc = StubEncoder()
    with SpatialMemory.open(tmp_path / "s3.smem", embedding_dim=DIM) as mem:  # no encoder
        mem.add_detections([_det(enc, "coffee mug", (0, 0, 0))])
        mem.commit()
        hits = mem.semantic("mug")
        assert hits and hits[0].label == "coffee mug"


def test_encoder_dim_mismatch_rejected(tmp_path) -> None:
    enc = StubEncoder()  # dim=16
    with pytest.raises(SchemaMismatchError):
        SpatialMemory.open(tmp_path / "s4.smem", embedding_dim=32, encoder=enc)
