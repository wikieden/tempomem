"""Pure-numpy 3D geometry helpers for perception adapters.

Schema-independent math used to lift camera-frame oriented 3D boxes (as a
`Cosmos3PerceptionAdapter` or any detector produces) into the world-frame
axis-aligned bboxes a `Detection` carries.

Conventions: right-handed, meters. A ``pose`` is a 4x4 homogeneous
**camera->world** transform. Orientation ``rpy`` = (roll, pitch, yaw) radians,
applied as ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (intrinsic Z-Y-X).
"""

from __future__ import annotations

import numpy as np

Vec3 = tuple[float, float, float]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) radians -> 3x3 rotation, R = Rz @ Ry @ Rx."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def transform_points(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points -> (N, 3)."""
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N, 4)
    out = h @ pose.T  # (N, 4)
    w = out[:, 3:4]
    w[w == 0] = 1.0
    return out[:, :3] / w


def transform_point(pose: np.ndarray, p: Vec3) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a single point -> (3,)."""
    return transform_points(pose, np.asarray(p, dtype=np.float64).reshape(1, 3))[0]


def oriented_box_corners(center: Vec3, size: Vec3, rpy: Vec3) -> np.ndarray:
    """8 corners of an oriented box in its own (parent) frame -> (8, 3).

    `size` is the full extent (length, width, height) along the box's local
    axes; `rpy` orients those axes; `center` places them.
    """
    half = np.asarray(size, dtype=np.float64).reshape(3) / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    local = signs * half  # (8, 3)
    rot = rpy_to_matrix(*(float(v) for v in rpy))
    return (local @ rot.T) + np.asarray(center, dtype=np.float64).reshape(3)


def world_aabb_from_obb(pose: np.ndarray, center: Vec3, size: Vec3, rpy: Vec3) -> tuple[Vec3, Vec3]:
    """Camera-frame oriented box -> world-frame axis-aligned bbox (min, max).

    Transforms the box's 8 corners by `pose` and takes the component-wise
    min/max — the AABB a `Detection.bbox_min` / `bbox_max` expects.
    """
    corners_world = transform_points(pose, oriented_box_corners(center, size, rpy))
    lo = corners_world.min(axis=0)
    hi = corners_world.max(axis=0)
    return (
        (float(lo[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(hi[1]), float(hi[2])),
    )
