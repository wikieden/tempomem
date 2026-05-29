from __future__ import annotations

import numpy as np
import pytest

from spatialmem import BadDetectionError, Detection


def test_feature_normalized() -> None:
    d = Detection(
        label="mug",
        feature=np.array([3.0, 4.0], dtype="float32"),
        center_xyz=(0, 0, 0),
        bbox_min=(0, 0, 0),
        bbox_max=(1, 1, 1),
    )
    assert pytest.approx(float(np.linalg.norm(d.feature)), abs=1e-5) == 1.0
    assert d.dim == 2


def test_bad_confidence() -> None:
    with pytest.raises(BadDetectionError):
        Detection("x", np.ones(4, "float32"), (0, 0, 0), (0, 0, 0), (1, 1, 1), confidence=2.0)


def test_zero_feature_rejected() -> None:
    with pytest.raises(BadDetectionError):
        Detection("x", np.zeros(4, "float32"), (0, 0, 0), (0, 0, 0), (1, 1, 1))


def test_bad_bbox() -> None:
    with pytest.raises(BadDetectionError):
        Detection("x", np.ones(4, "float32"), (0, 0, 0), (1, 1, 1), (0, 0, 0))


def test_json_roundtrip() -> None:
    d = Detection(
        "mug",
        np.array([1.0, 2.0, 2.0], "float32"),
        (1, 2, 3),
        (0, 0, 0),
        (2, 4, 6),
        confidence=0.7,
        ts=123.0,
        aux={"k": "v"},
    )
    d2 = Detection.from_json(d.to_json())
    assert d2.label == d.label
    assert d2.confidence == d.confidence
    assert d2.ts == d.ts
    assert d2.aux == d.aux
    assert np.allclose(d2.feature, d.feature)
