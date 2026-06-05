from __future__ import annotations

import numpy as np

from spatialmem.geometry import (
    oriented_box_corners,
    rpy_to_matrix,
    transform_point,
    transform_points,
    world_aabb_from_obb,
)

I4 = np.eye(4)


def _pose(translation=(0.0, 0.0, 0.0), rot=None) -> np.ndarray:
    p = np.eye(4)
    if rot is not None:
        p[:3, :3] = rot
    p[:3, 3] = translation
    return p


def test_transform_point_identity() -> None:
    out = transform_point(I4, (1.0, 2.0, 3.0))
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_transform_point_translation() -> None:
    out = transform_point(_pose(translation=(10.0, -1.0, 0.5)), (1.0, 2.0, 3.0))
    assert np.allclose(out, [11.0, 1.0, 3.5])


def test_transform_points_batch() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = transform_points(_pose(translation=(1.0, 2.0, 3.0)), pts)
    assert out.shape == (2, 3)
    assert np.allclose(out, [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])


def test_rpy_yaw_90_rotates_x_to_y() -> None:
    rot = rpy_to_matrix(0.0, 0.0, np.pi / 2)
    out = rot @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_rpy_matrix_is_orthonormal() -> None:
    rot = rpy_to_matrix(0.3, -0.7, 1.1)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_oriented_box_corners_count_and_extent() -> None:
    corners = oriented_box_corners((0.0, 0.0, 0.0), (2.0, 4.0, 6.0), (0.0, 0.0, 0.0))
    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), [-1.0, -2.0, -3.0])
    assert np.allclose(corners.max(axis=0), [1.0, 2.0, 3.0])


def test_world_aabb_axis_aligned_identity() -> None:
    lo, hi = world_aabb_from_obb(I4, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    assert np.allclose(lo, [-1.0, -1.0, -1.0])
    assert np.allclose(hi, [1.0, 1.0, 1.0])


def test_world_aabb_with_pose_translation() -> None:
    lo, hi = world_aabb_from_obb(
        _pose(translation=(5.0, 0.0, 1.0)), (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0)
    )
    assert np.allclose(lo, [4.0, -1.0, 0.0])
    assert np.allclose(hi, [6.0, 1.0, 2.0])


def test_world_aabb_grows_under_yaw_rotation() -> None:
    # a 2x2x2 box rotated 45 deg about z: in-plane half-extent -> sqrt(2)
    _lo, hi = world_aabb_from_obb(I4, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (0.0, 0.0, np.pi / 4))
    assert np.isclose(hi[0], np.sqrt(2.0), atol=1e-9)
    assert np.isclose(hi[1], np.sqrt(2.0), atol=1e-9)
    assert np.isclose(hi[2], 1.0, atol=1e-9)  # z unchanged by yaw
