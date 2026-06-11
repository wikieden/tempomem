"""Value objects for ingest. See spec/API.md and spec/SCHEMA.md.

Coordinates: right-handed, meters, world frame.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ._errors import BadDetectionError

Vec3 = tuple[float, float, float]


def _as_f32(feature: Any) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise BadDetectionError("feature vector is empty")
    n = float(np.linalg.norm(arr))
    if n == 0.0 or not np.isfinite(n):
        raise BadDetectionError("feature vector has zero or non-finite norm")
    return arr / n


@dataclass(frozen=True, slots=True)
class Detection:
    """A single open-vocabulary detection in world coordinates.

    feature is L2-normalized on construction. center/bbox are meters.
    """

    label: str
    feature: np.ndarray
    center_xyz: Vec3
    bbox_min: Vec3
    bbox_max: Vec3
    confidence: float = 1.0
    mask_rle: bytes | None = None
    ts: float | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature", _as_f32(self.feature))
        if not (0.0 <= self.confidence <= 1.0):
            raise BadDetectionError(f"confidence {self.confidence} not in [0,1]")
        if not self.label:
            raise BadDetectionError("label must be non-empty")
        for name in ("center_xyz", "bbox_min", "bbox_max"):
            v = getattr(self, name)
            if len(v) != 3:
                raise BadDetectionError(f"{name} must have 3 components")
        lo, hi = self.bbox_min, self.bbox_max
        if any(hi[i] < lo[i] for i in range(3)):
            raise BadDetectionError("bbox_max must be >= bbox_min componentwise")

    @property
    def dim(self) -> int:
        return int(self.feature.shape[0])

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature"] = self.feature.tolist()
        if self.mask_rle is not None:
            d["mask_rle"] = base64.b64encode(self.mask_rle).decode("ascii")
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Detection:
        mask = d.get("mask_rle")
        return cls(
            label=d["label"],
            feature=np.asarray(d["feature"], dtype=np.float32),
            center_xyz=tuple(d["center_xyz"]),  # type: ignore[arg-type]
            bbox_min=tuple(d["bbox_min"]),  # type: ignore[arg-type]
            bbox_max=tuple(d["bbox_max"]),  # type: ignore[arg-type]
            confidence=d.get("confidence", 1.0),
            mask_rle=base64.b64decode(mask) if isinstance(mask, str) else None,
            ts=d.get("ts"),
            aux=d.get("aux", {}),
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """A persisted detection with assigned id + episode + resolved timestamp."""

    id: int
    episode_id: int
    ts: float
    label: str
    confidence: float
    center_xyz: Vec3
    bbox_min: Vec3
    bbox_max: Vec3
    feature: np.ndarray
