"""Text/image encoders for semantic queries.

Chronotope is BYO-encoder: features on `Detection` come from the user's
perception model; to run a natural-language `query("red mug")` the same
embedding space must be reachable for the query string. Supply any object
implementing the `Encoder` protocol to `TempoMem.open(encoder=...)`.

`OpenClipEncoder` is an optional reference implementation behind the `[clip]`
extra. The core package depends on numpy only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class Encoder(Protocol):
    """Maps text (and optionally images) into the store's embedding space.

    `encode_text` must return an (N, dim) float32 array, L2-normalized per row,
    matching the store's `embedding_dim`.
    """

    @property
    def dim(self) -> int: ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class OpenClipEncoder:
    """Reference CLIP encoder. Requires `pip install tempomem[clip]`.

    Lazy-imports open_clip/torch so importing this module never pulls Torch.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        try:
            # reason: optional [clip] extra; guarded by this try/except
            import open_clip  # pyright: ignore[reportMissingImports]
            import torch  # pyright: ignore[reportMissingImports]
        except ImportError as e:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OpenClipEncoder needs the 'clip' extra: pip install tempomem[clip]"
            ) from e
        self._torch = torch
        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._device = device
        self._dim = int(self._model.text_projection.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs torch
        torch = self._torch
        toks = self._tokenizer(list(texts)).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(toks)
        return l2_normalize(feats.cpu().numpy())
