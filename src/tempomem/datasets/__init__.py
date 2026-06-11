"""Dataset sources — stream per-frame ground-truth detections into a store.

The product pitch is incremental fusion: the same object seen across many
frames converges to one node. This module provides that stream without any
perception model or GPU. `SyntheticScene` generates a deterministic multi-frame
scene; a real `ReplicaAdapter` / `ScanNetAdapter` (parsing GT instance masks +
depth) plugs into the same `DatasetSource` shape later. See docs/DEV-PLAN.md.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
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
    use `tempomem.encoders.OpenClipEncoder` (the `[clip]` extra) for that.
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


def gt_detections_from_frame(
    depth: np.ndarray,
    instance: np.ndarray,
    pose: np.ndarray,
    *,
    intrinsics: tuple[float, float, float, float],
    labels: dict[int, str],
    encoder: HashEncoder,
    ts: float = 0.0,
    min_pixels: int = 1,
    drop_ids: tuple[int, ...] = (0,),
) -> list[Detection]:
    """Ground-truth instance masks + depth -> world-frame ``Detection``s.

    Pure numpy, no model, no GPU. For each instance id in ``instance`` (a per-
    pixel ``(H, W)`` integer id map), the masked depth pixels are deprojected to
    camera-frame 3D points via pinhole ``intrinsics`` ``(fx, fy, cx, cy)`` —
    OpenCV convention: +x right, +y down, +z forward — transformed to the world
    frame by the 4x4 camera->world ``pose``, then reduced to a centroid + axis-
    aligned bbox (meters). ``labels`` maps instance id -> label; ids in
    ``drop_ids`` (default background ``0``) and instances with fewer than
    ``min_pixels`` valid depth pixels are skipped.
    """
    fx, fy, cx, cy = intrinsics
    depth = np.asarray(depth, dtype=np.float64)
    instance = np.asarray(instance).astype(np.int64)
    p = np.asarray(pose, dtype=np.float64)
    rot, trans = p[:3, :3], p[:3, 3]
    dets: list[Detection] = []
    for iid in (int(i) for i in np.unique(instance)):
        if iid in drop_ids:
            continue
        sel = (instance == iid) & (depth > 0)
        vs, us = np.where(sel)  # np.where -> (rows=v, cols=u); us drives x, vs drives y
        if vs.size < min_pixels:
            continue
        d = depth[vs, us]
        cam = np.stack([(us - cx) * d / fx, (vs - cy) * d / fy, d], axis=1)
        world = cam @ rot.T + trans
        cmin, cmax, ctr = world.min(0), world.max(0), world.mean(0)
        label = labels.get(iid, str(iid))
        dets.append(
            Detection(
                label=label,
                feature=encoder.encode_text([label])[0],
                center_xyz=(float(ctr[0]), float(ctr[1]), float(ctr[2])),
                bbox_min=(float(cmin[0]), float(cmin[1]), float(cmin[2])),
                bbox_max=(float(cmax[0]), float(cmax[1]), float(cmax[2])),
                confidence=1.0,
                ts=ts,
                aux={"source": "replica-gt", "instance_id": iid},
            )
        )
    return dets


@dataclass
class ReplicaAdapter:
    """Stream ground-truth Replica detections as a ``DatasetSource``.

    ``reader`` yields ``(depth, instance, pose)`` per frame — inject a list of
    arrays for tests, or use :class:`ReplicaFileReader` to read a real scene
    directory. Each frame becomes world-frame detections via
    :func:`gt_detections_from_frame`. The whole path is pure numpy (no model, no
    GPU): GT annotations in, fused nodes out.
    """

    reader: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]
    intrinsics: tuple[float, float, float, float]
    labels: dict[int, str]
    encoder: HashEncoder
    min_pixels: int = 50
    drop_ids: tuple[int, ...] = (0,)

    def frames(self) -> Iterator[list[Detection]]:
        for i, (depth, instance, pose) in enumerate(self.reader):
            yield gt_detections_from_frame(
                depth,
                instance,
                pose,
                intrinsics=self.intrinsics,
                labels=self.labels,
                encoder=self.encoder,
                ts=float(i),
                min_pixels=self.min_pixels,
                drop_ids=self.drop_ids,
            )


@dataclass
class ReplicaFileReader:
    """Read a Nice-SLAM / ConceptGraphs-style Replica scene directory.

    Assumed layout (**verify against your Replica variant** — exact filenames
    and the depth scale differ between releases):

        ``scene_dir/results/depth{i:06d}.png``     16-bit depth / ``depth_scale`` -> meters
        ``scene_dir/results/instance{i:06d}.png``  per-pixel instance ids
        ``scene_dir/traj.txt``                      one row-major 4x4 camera->world pose per line

    Needs the ``[replica]`` extra (imageio). This file-I/O path is **not**
    exercised in CI (no dataset shipped in the repo); the geometry it feeds
    (:func:`gt_detections_from_frame`) is unit-tested. Iterating yields
    ``(depth_m, instance, pose)`` tuples ready for :class:`ReplicaAdapter`.
    """

    scene_dir: str
    depth_scale: float = 6553.5
    stride: int = 1
    depth_pattern: str = "results/depth{i:06d}.png"
    instance_pattern: str = "results/instance{i:06d}.png"
    traj_name: str = "traj.txt"

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        import os

        iio = _import_imageio()
        poses = self._load_traj()
        for i in range(0, len(poses), self.stride):
            depth = (
                np.asarray(iio.imread(os.path.join(self.scene_dir, self.depth_pattern.format(i=i))))
                / self.depth_scale
            )
            instance = np.asarray(
                iio.imread(os.path.join(self.scene_dir, self.instance_pattern.format(i=i)))
            )
            yield depth, instance, poses[i]

    def _load_traj(self) -> np.ndarray:
        import os

        rows = np.loadtxt(os.path.join(self.scene_dir, self.traj_name))
        return rows.reshape(-1, 4, 4)


def _import_imageio():  # pragma: no cover - thin optional-dep shim
    try:
        import imageio.v3 as iio  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError(
            "ReplicaFileReader needs the [replica] extra: pip install 'tempomem[replica]'"
        ) from e
    return iio
