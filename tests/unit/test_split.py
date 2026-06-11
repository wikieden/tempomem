from __future__ import annotations

from tempomem import ChronotopeConfig, FusionConfig, SpatialMemory
from tests.conftest import DIM, make_det


def _merge_all_cfg() -> ChronotopeConfig:
    # tau_merge low + huge search dilation -> identical-feature dets merge into
    # one node regardless of distance, producing a two-cluster node to split.
    return ChronotopeConfig(
        fusion=FusionConfig(
            search_dilation_m=100.0,
            dist_norm_m=100.0,
            tau_merge=0.05,
            tau_split_m=1.0,
            min_split_obs=2,
        )
    )


def test_resplit_separates_two_clusters(tmp_path) -> None:
    with SpatialMemory.open(
        tmp_path / "sp.smem", embedding_dim=DIM, config=_merge_all_cfg()
    ) as mem:
        # same label+seed (identical feature) at two distant clusters
        mem.add_detections(
            [
                make_det("mug", (0.0, 0.0, 0.0), 1),
                make_det("mug", (0.1, 0.0, 0.0), 1),
                make_det("mug", (10.0, 0.0, 0.0), 1),
                make_det("mug", (10.1, 0.0, 0.0), 1),
            ]
        )
        mem.commit()
        assert mem.stats().n_nodes == 1  # all merged into one (two clusters)

        split, created = mem.resplit()
        assert split == 1
        assert created == 2
        assert mem.stats().n_nodes == 2

        centroids = sorted(n.center_xyz[0] for n in mem.recent(n=10))
        assert centroids[0] < 1.0  # near-origin cluster
        assert centroids[1] > 9.0  # far cluster


def test_resplit_no_op_for_tight_node(mem) -> None:
    # default config: two nearby sightings of one object -> one tight node
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.0), 1), make_det("mug", (1.02, 0.0, 0.0), 1)])
    mem.commit()
    before = mem.stats().n_nodes
    split, created = mem.resplit()
    assert split == 0
    assert created == 0
    assert mem.stats().n_nodes == before


def test_resplit_skips_too_few_obs(mem) -> None:
    mem.add_detections([make_det("mug", (0, 0, 0), 1)])
    mem.commit()
    assert mem.resplit() == (0, 0)
