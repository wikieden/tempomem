from __future__ import annotations

import pytest

from spatialmem import SpatialMemTools, ToolError
from tests.conftest import make_det

EXPECTED = {
    "semantic_search",
    "spatial_query",
    "whats_in",
    "whats_on",
    "recent_changes",
    "serialize_scene",
}


def _seed(mem):
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1), make_det("table", (1.0, 0.0, 0.4), 2)])
    mem.commit()


def test_schemas_shape(mem) -> None:
    tools = SpatialMemTools(mem)
    schemas = tools.schemas()
    assert {s["name"] for s in schemas} == EXPECTED
    for s in schemas:
        assert "description" in s
        assert s["parameters"]["type"] == "object"
    assert set(tools.names) == EXPECTED


def test_semantic_search_envelope(mem) -> None:
    _seed(mem)
    out = SpatialMemTools(mem).call("semantic_search", {"text": "mug"})
    assert out["hits"], "expected at least one hit"
    h = out["hits"][0]
    assert set(h) == {"node_id", "label", "centroid_m", "confidence", "score", "t_last"}
    assert isinstance(h["node_id"], int)
    assert len(h["centroid_m"]) == 3


def test_spatial_query(mem) -> None:
    _seed(mem)
    out = SpatialMemTools(mem).call("spatial_query", {"near": [1.0, 0.0, 0.9], "radius_m": 1.0})
    labels = {h["label"] for h in out["hits"]}
    assert "mug" in labels


def test_whats_in_region(mem) -> None:
    _seed(mem)
    mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    out = SpatialMemTools(mem).call("whats_in", {"region": "kitchen"})
    labels = {h["label"] for h in out["hits"]}
    assert "mug" in labels


def test_whats_on_structure(mem) -> None:
    _seed(mem)
    mem.relate()
    out = SpatialMemTools(mem).call("whats_on", {"anchor": "table"})
    assert "hits" in out and isinstance(out["hits"], list)
    assert "meta" in out  # echoes resolved anchor / relation context


def test_recent_changes_split(mem) -> None:
    _seed(mem)  # observations at ts=1000
    tools = SpatialMemTools(mem)
    out = tools.call("recent_changes", {"since_ts": 999.0})
    assert out["new"], "nodes first seen at 1000 are new since 999"
    later = tools.call("recent_changes", {"since_ts": 2000.0})
    assert later["new"] == [] and later["seen_again"] == []


def test_serialize_scene(mem) -> None:
    _seed(mem)
    out = SpatialMemTools(mem).call("serialize_scene", {"max_tokens": 200})
    assert isinstance(out["scene"], str) and out["scene"]


def test_unknown_tool_raises(mem) -> None:
    with pytest.raises(ToolError):
        SpatialMemTools(mem).call("nope")


def test_missing_required_arg_raises(mem) -> None:
    with pytest.raises(ToolError):
        SpatialMemTools(mem).call("semantic_search", {})


def test_bad_near_raises(mem) -> None:
    with pytest.raises(ToolError):
        SpatialMemTools(mem).call("spatial_query", {"near": [1.0, 0.0]})


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("semantic_search", {"text": "mug", "k": "abc"}),  # non-int k
        ("semantic_search", {"text": "mug", "k": 0}),  # k below range
        ("semantic_search", {"text": "mug", "k": True}),  # bool k
        ("spatial_query", {"near": [float("nan"), 0.0, 0.0]}),  # non-finite
        ("spatial_query", {"near": [1.0, 0.0, 0.0], "radius_m": -1.0}),  # radius <= 0
        ("spatial_query", {"near": [1.0, 0.0, 0.0], "radius_m": "wide"}),  # non-number
        ("serialize_scene", {"max_tokens": 0}),  # below range (0 != "no budget")
        ("recent_changes", {"since_ts": "oops"}),  # non-number
    ],
)
def test_malformed_args_raise_toolerror(mem, name, args) -> None:
    with pytest.raises(ToolError):
        SpatialMemTools(mem).call(name, args)


def test_serialize_scene_no_budget_ok(mem) -> None:
    _seed(mem)
    out = SpatialMemTools(mem).call("serialize_scene", {})  # absent max_tokens = full scene
    assert isinstance(out["scene"], str) and out["scene"]
