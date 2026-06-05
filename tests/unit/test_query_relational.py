from __future__ import annotations

from tests.conftest import make_det


def _scene(mem) -> None:
    mem.add_detections(
        [
            make_det("table", (1.0, 0.0, 0.40), 1),
            make_det("book", (1.0, 0.0, 0.55), 2),  # on the table
            make_det("lamp", (1.35, 0.0, 0.45), 3),  # near the table, not on it
        ]
    )
    mem.commit()
    mem.relate()


def test_query_on_relation(mem) -> None:
    _scene(mem)
    res = mem.query("what's on the table")
    assert res.debug.get("relation") == "on"
    assert res.debug.get("anchor") == "table"
    assert [h.label for h in res.nodes] == ["book"]


def test_query_near_relation(mem) -> None:
    _scene(mem)
    res = mem.query("what is near the table")
    assert res.debug.get("relation") == "near"
    assert "lamp" in {h.label for h in res.nodes}


def test_query_relation_no_anchor_falls_back(mem) -> None:
    _scene(mem)
    res = mem.query("near the spaceship")  # no such object → fall back
    assert res.debug.get("relation") is None


def test_query_no_relation_phrase(mem) -> None:
    _scene(mem)
    res = mem.query("book")  # plain keyword, no relation
    assert res.debug.get("relation") is None
    assert res.nodes and res.nodes[0].label == "book"
