"""Dataset sources — stream per-frame ground-truth detections into a store.

The product pitch is incremental fusion: the same object seen across many
frames converges to one node. This module provides that stream without any
perception model or GPU. `SyntheticScene` generates a deterministic multi-frame
scene; a real `ReplicaAdapter` / `ScanNetAdapter` (parsing GT instance masks +
depth) plugs into the same `DatasetSource` shape later. See docs/DEV-PLAN.md.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..frame import Detection

if TYPE_CHECKING:
    from .. import SpatialMemory

Vec3 = tuple[float, float, float]


@runtime_checkable
class DatasetSource(Protocol):
    """Yields one list of world-frame ground-truth detections per frame."""

    def frames(self) -> Iterator[list[Detection]]: ...


class HashEncoder:
    """Deterministic fixture encoder: text -> seeded unit vector.

    For demos/tests only — same string always maps to the same vector, so a
    detection labelled "mug" and the query "mug" align. NOT real semantics;
    use `spatialmem.encoders.OpenClipEncoder` (the `[clip]` extra) for that.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**32)
            v = np.random.default_rng(seed).standard_normal(self._dim).astype("float32")
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype="float32")


@dataclass
class SyntheticScene:
    """A deterministic multi-frame scene: fixed objects, observed each frame
    with small positional jitter (simulating a moving camera). Streaming this
    through fusion converges each object to a single node.
    """

    objects: list[tuple[str, Vec3]]  # (label, world center)
    encoder: HashEncoder
    n_frames: int = 12
    noise_m: float = 0.02
    half_extent_m: float = 0.05
    seed: int = 0
    _feats: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def _feat(self, label: str) -> np.ndarray:
        if label not in self._feats:
            self._feats[label] = self.encoder.encode_text([label])[0]
        return self._feats[label]

    def frames(self) -> Iterator[list[Detection]]:
        rng = np.random.default_rng(self.seed)
        h = self.half_extent_m
        for f in range(self.n_frames):
            dets: list[Detection] = []
            for label, center in self.objects:
                jit = rng.normal(0.0, self.noise_m, 3)
                c = (center[0] + jit[0], center[1] + jit[1], center[2] + jit[2])
                dets.append(
                    Detection(
                        label=label,
                        feature=self._feat(label),
                        center_xyz=c,
                        bbox_min=(c[0] - h, c[1] - h, c[2] - h),
                        bbox_max=(c[0] + h, c[1] + h, c[2] + h),
                        confidence=0.9,
                        ts=float(f),
                    )
                )
            yield dets


def stream(
    mem: SpatialMemory,
    source: DatasetSource,
    *,
    commit_every: int = 1,
    episode: str | None = None,
) -> tuple[int, int]:
    """Ingest every frame's detections into `mem`. Returns (frames, observations)."""
    n_frames = 0
    n_obs = 0
    for i, dets in enumerate(source.frames(), start=1):
        mem.add_detections(dets, episode=episode)
        n_frames += 1
        n_obs += len(dets)
        if commit_every and i % commit_every == 0:
            mem.commit()
    mem.commit()
    return n_frames, n_obs
