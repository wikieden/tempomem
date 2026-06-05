"""Text/image encoders for semantic queries.

SpatialMem is BYO-encoder: features on `Detection` come from the user's
perception model; to run a natural-language `query("red mug")` the same
embedding space must be reachable for the query string. Supply any object
implementing the `Encoder` protocol to `SpatialMemory.open(encoder=...)`.

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


@runtime_checkable
class ImageEncoder(Protocol):
    """Maps image crops into the store's embedding space.

    A perception adapter that only emits boxes (e.g. `Cosmos3PerceptionAdapter`)
    encodes each object's image crop to get the per-object feature `Detection`
    requires. `encode_image` returns an (N, dim) float32 array, L2-normalized
    per row, in the SAME space as the encoder's `encode_text` so semantic query
    still aligns. Images are (H, W, 3) uint8 RGB arrays.
    """

    @property
    def dim(self) -> int: ...

    def encode_image(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class OpenClipEncoder:
    """Reference CLIP encoder. Requires `pip install spatialmem[clip]`.

    Lazy-imports open_clip/torch so importing this module never pulls Torch.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as e:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OpenClipEncoder needs the 'clip' extra: pip install spatialmem[clip]"
            ) from e
        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
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

    def encode_image(self, images: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
        from PIL import Image

        torch = self._torch
        batch = torch.stack(
            [self._preprocess(Image.fromarray(np.asarray(im, dtype=np.uint8))) for im in images]
        ).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
        return l2_normalize(feats.cpu().numpy())
