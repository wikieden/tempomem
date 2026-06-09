"""Eval-suite v0: recall@k, restart/cross-episode persistence, decay/forget."""

from __future__ import annotations

from spatialmem import SpatialMemory
from spatialmem.bench import decay_forget, persistence_after_reopen, recall_at_k
from tests.conftest import DIM, make_det


def _two_episode_kitchen(mem) -> None:
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)], episode="shift-1")
    mem.add_detections([make_det("kettle", (3.0, 0.0, 0.9), 2)], episode="shift-2")
    mem.commit()


def test_recall_at_k_baseline(tmp_path) -> None:
    with SpatialMemory.open(tmp_path / "r.smem", embedding_dim=DIM) as mem:
        _two_episode_kitchen(mem)
        rep = recall_at_k(mem, [("mug", "mug"), ("kettle", "kettle")], k=5)
    assert rep.recall == 1.0
    assert rep.misses == []


def test_persistence_across_reopen_and_episodes(tmp_path) -> None:
    path = tmp_path / "p.smem"
    with SpatialMemory.open(path, embedding_dim=DIM) as mem:
        _two_episode_kitchen(mem)  # objects ingested under two episodes, then closed
    # reopen ("restart") and confirm both episodes' objects are still retrievable
    rep = persistence_after_reopen(
        str(path), embedding_dim=DIM, cases=[("mug", "mug"), ("kettle", "kettle")]
    )
    assert rep.recall == 1.0


def test_decay_forget_lifecycle(tmp_path) -> None:
    with SpatialMemory.open(tmp_path / "d.smem", embedding_dim=DIM) as mem:
        _two_episode_kitchen(mem)
        # age ~1 year with a 1-day half-life: confidence collapses below the floor
        far_future = 1000.0 + 86400.0 * 365
        rep = decay_forget(mem, half_life_days=1.0, min_conf=0.5, now=far_future)
        assert rep.nodes_before == 2
        assert rep.pruned == 2  # both aged-out nodes pruned
        assert rep.nodes_after_decay == 0
