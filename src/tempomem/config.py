"""Tunable configuration. See spec/FUSION-ARBITER.md for the meaning of each
fusion threshold. All defaults tuned on the synthetic kitchen set.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FusionConfig:
    # candidate search
    search_dilation_m: float = 0.25  # bbox dilation for proximity query
    # geometry
    dist_norm_m: float = 0.50  # distance at which s_geom hits 0
    # score weights (sum to 1.0)
    w_geom: float = 0.2
    w_iou: float = 0.2
    w_sem: float = 0.5
    w_label: float = 0.1
    # decision thresholds
    tau_merge: float = 0.62
    tau_ambig: float = 0.45
    tau_obs: float = 0.30  # reject observations below this confidence
    # merge dynamics
    centroid_alpha: float = 0.2  # EMA factor for centroid/feature
    conf_gain: float = 0.5  # how fast confidence saturates toward 1
    # split detection (M2): a node whose member observations form two clusters
    # separated by more than tau_split_m, each with >= min_split_obs members,
    # is split back into two nodes.
    tau_split_m: float = 1.0
    min_split_obs: int = 2

    def __post_init__(self) -> None:
        total = self.w_geom + self.w_iou + self.w_sem + self.w_label
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fusion weights must sum to 1.0, got {total}")


@dataclass(frozen=True, slots=True)
class ChronotopeConfig:
    fusion: FusionConfig = field(default_factory=FusionConfig)
    # Auto-flush threshold: if set, add_detections() calls commit() automatically
    # once _pending reaches this length, and emits a WARNING log line.
    # None (default) disables the limit — caller is responsible for commit().
    max_pending_obs: int | None = None

    def __post_init__(self) -> None:
        # spec: ENGINEERING.md §auto-flush — a threshold below 1 would flush after
        # every observation, which is a config mistake rather than a valid mode.
        if self.max_pending_obs is not None and self.max_pending_obs < 1:
            raise ValueError(f"max_pending_obs must be >= 1 or None, got {self.max_pending_obs}")
