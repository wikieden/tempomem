"""OpenClipEncoder smoke test — runs only when the [clip] extra is installed.

Uses pretrained=None (random init) so CI validates the API shape/dim without
downloading model weights. Skipped entirely when torch/open_clip are absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("open_clip")

from spatialmem.encoders import OpenClipEncoder


def test_openclip_encode_text_shape() -> None:
    enc = OpenClipEncoder(model_name="ViT-B-32", pretrained=None)
    v = enc.encode_text(["a red mug", "a kettle"])
    assert v.shape == (2, enc.dim)
    norms = (v**2).sum(axis=1) ** 0.5
    assert all(abs(float(n) - 1.0) < 1e-3 for n in norms)
