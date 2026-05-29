from __future__ import annotations

import numpy as np
import pytest

from spatialmem import Detection, SpatialMemory

DIM = 16


@pytest.fixture
def dim() -> int:
    return DIM


def make_det(label: str, center, seed: int, ts: float = 1000.0) -> Detection:
    rng = np.random.default_rng(seed)
    cx, cy, cz = center
    return Detection(
        label=label,
        feature=rng.standard_normal(DIM).astype("float32"),
        center_xyz=center,
        bbox_min=(cx - 0.05, cy - 0.05, cz - 0.05),
        bbox_max=(cx + 0.05, cy + 0.05, cz + 0.05),
        confidence=0.9,
        ts=ts,
    )


@pytest.fixture
def mem(tmp_path):
    m = SpatialMemory.open(tmp_path / "t.smem", embedding_dim=DIM)
    yield m
    m.close()
