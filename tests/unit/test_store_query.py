from __future__ import annotations

import json

import pytest

from spatialmem import SchemaMismatchError, SpatialMemory
from tests.conftest import DIM, make_det


def test_ingest_and_stats(mem) -> None:
    ids = mem.add_detections([make_det("mug", (1, 0, 0), 1), make_det("sink", (0, 1, 0), 2)])
    assert len(ids) == 2
    cs = mem.commit()
    assert cs.observations_committed == 2
    st = mem.stats()
    assert st.n_nodes == 2  # M0 stub: one node per obs
    assert st.n_obs == 2
    assert st.n_episodes == 1


def test_query_semantic_keyword(mem) -> None:
    mem.add_detections([make_det("coffee mug", (1, 0, 0), 1), make_det("kettle", (2, 0, 0), 2)])
    mem.commit()
    res = mem.query("mug")
    assert res.nodes
    assert res.nodes[0].label == "coffee mug"


def test_query_temporal(mem) -> None:
    mem.add_detections([make_det("a", (0, 0, 0), 1, ts=10.0)])
    mem.add_detections([make_det("b", (0, 0, 0), 2, ts=20.0)])
    mem.commit()
    res = mem.query("what did I see recently")
    assert res.intent_used == "temporal"
    assert res.nodes[0].label == "b"  # newest first


def test_spatial_near(mem) -> None:
    mem.add_detections([make_det("near", (0.1, 0, 0), 1), make_det("far", (5, 0, 0), 2)])
    mem.commit()
    hits = mem.spatial(near=(0, 0, 0), radius=1.0)
    assert [h.label for h in hits] == ["near"]


def test_forget(mem) -> None:
    mem.add_detections([make_det("mug", (1, 0, 0), 1)])
    mem.commit()
    node = mem.recent(n=1)[0]
    mem.forget(node.id)
    assert mem.stats().n_nodes == 0


def test_serialize_json_roundtrips_counts(mem) -> None:
    mem.add_detections([make_det("mug", (1, 0, 0), 1)])
    mem.commit()
    data = json.loads(mem.serialize(format="json"))
    assert data["schema_version"] == 1
    assert data["embedding_dim"] == DIM
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["label"] == "mug"


def test_serialize_prompt(mem) -> None:
    mem.add_detections([make_det("mug", (1.2, 0.3, 0.9), 1)])
    mem.commit()
    txt = mem.serialize(format="prompt")
    assert "SCENE" in txt
    assert '"mug"' in txt


def test_persistence_across_reopen(tmp_path) -> None:
    p = tmp_path / "persist.smem"
    with SpatialMemory.open(p, embedding_dim=DIM) as m:
        m.add_detections([make_det("mug", (1, 0, 0), 1)])
        m.commit()
    with SpatialMemory.open(p, embedding_dim=DIM) as m2:
        assert m2.stats().n_nodes == 1


def test_dim_mismatch_on_reopen(tmp_path) -> None:
    p = tmp_path / "dim.smem"
    with SpatialMemory.open(p, embedding_dim=DIM) as m:
        m.add_detections([make_det("mug", (1, 0, 0), 1)])
        m.commit()
    with pytest.raises(SchemaMismatchError):
        SpatialMemory.open(p, embedding_dim=DIM + 1)
