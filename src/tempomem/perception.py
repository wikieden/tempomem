"""Perception adapter seam: RGB-D frame -> open-vocab 3D detections.

Chronotope's core is detections-in (BYO perception). `add_frame` routes a raw
RGB-D frame through a `PerceptionAdapter` to produce `Detection`s, then fuses
them. The first concrete adapter (ConceptGraphs: SAM + Grounding DINO +
OpenCLIP, Apache/MIT) lands behind a `[perception]` extra on a CUDA dev box —
see docs/sprint/SPRINT-02.md P1. The protocol below is the stable seam so any
backend (or a test stub) can plug in without touching the core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from .frame import Detection


@runtime_checkable
class PerceptionAdapter(Protocol):
    """Turns one posed RGB-D frame into world-frame open-vocab detections.

    - rgb: (H, W, 3) uint8
    - depth: (H, W) float32 meters
    - pose: (4, 4) camera-to-world transform
    - intrinsics: (3, 3) camera matrix, or None if the adapter has its own
    Returns a list of `Detection` in world coordinates (meters).
    """

    def process_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> list[Detection]: ...
