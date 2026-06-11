from __future__ import annotations

from tempomem import DatasetSource, HashEncoder, SpatialMemory, SyntheticScene, stream
from tempomem.bench import recall_at_k

DIM = 64

_OBJECTS = [
    ("mug", (1.0, 0.0, 1.0)),
    ("kettle", (1.5, 0.0, 1.0)),
    ("fridge", (3.0, 0.0, 1.2)),
    ("sink", (0.5, 0.0, 0.9)),
]


def _scene(enc: HashEncoder) -> SyntheticScene:
    return SyntheticScene(objects=list(_OBJECTS), encoder=enc, n_frames=12)


def test_source_satisfies_protocol() -> None:
    scene = SyntheticScene(objects=[("a", (0, 0, 0))], encoder=HashEncoder(8))
    assert isinstance(scene, DatasetSource)


def test_stream_dedups_to_one_node_per_object(tmp_path) -> None:
    enc = HashEncoder(DIM)
    with SpatialMemory.open(tmp_path / "s.smem", embedding_dim=DIM, encoder=enc) as mem:
        n_frames, n_obs = stream(mem, _scene(enc), commit_every=4)
        assert n_frames == 12
        assert n_obs == 48  # 12 frames x 4 objects
        # incremental fusion: 48 observations converge to 4 nodes
        assert mem.stats().n_nodes == 4  # 48 obs deduped to 4 objects
        assert mem.stats().n_obs == 48


def test_stream_then_query_recall(tmp_path) -> None:
    enc = HashEncoder(DIM)
    with SpatialMemory.open(tmp_path / "s2.smem", embedding_dim=DIM, encoder=enc) as mem:
        stream(mem, _scene(enc))
        cases = [(lbl, lbl) for lbl, _ in _OBJECTS]
        rep = recall_at_k(mem, cases, k=3)
        assert rep.recall >= 0.8  # M2 exit metric proxy (5-question style)


def test_stream_deterministic(tmp_path) -> None:
    enc = HashEncoder(DIM)
    counts = []
    for name in ("a", "b"):
        with SpatialMemory.open(tmp_path / f"{name}.smem", embedding_dim=DIM, encoder=enc) as mem:
            stream(mem, _scene(enc))
            counts.append(mem.stats().n_nodes)
    assert counts[0] == counts[1] == 4
