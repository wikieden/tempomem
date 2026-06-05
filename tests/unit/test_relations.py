from __future__ import annotations

from tests.conftest import make_det


def test_relate_on_and_under(mem) -> None:
    # book resting on a table (xy overlap, book bottom ~ table top, book higher)
    mem.add_detections(
        [make_det("table", (1.0, 0.0, 0.40), 1), make_det("book", (1.0, 0.0, 0.55), 2)]
    )
    mem.commit()
    assert mem.stats().n_nodes == 2  # stay distinct
    written = mem.relate()
    assert written > 0

    on = [h.label for h, t in mem.related("book") if t == "on"]
    assert "table" in on  # book on table
    under = [h.label for h, t in mem.related("table") if t == "under"]
    assert "book" in under  # table under book


def test_relate_near_symmetric(mem) -> None:
    mem.add_detections(
        [make_det("mug", (1.0, 0.0, 0.9), 1), make_det("kettle", (1.3, 0.0, 0.9), 2)]
    )
    mem.commit()
    mem.relate(near_m=0.6)
    assert "kettle" in [h.label for h, t in mem.related("mug", rel="near")]
    assert "mug" in [h.label for h, t in mem.related("kettle", rel="near")]


def test_relate_far_objects_not_near(mem) -> None:
    mem.add_detections([make_det("mug", (0.0, 0.0, 0.9), 1), make_det("door", (9.0, 0.0, 1.0), 2)])
    mem.commit()
    mem.relate(near_m=0.6)
    assert mem.related("mug", rel="near") == []


def test_relate_idempotent(mem) -> None:
    mem.add_detections(
        [make_det("mug", (1.0, 0.0, 0.9), 1), make_det("kettle", (1.3, 0.0, 0.9), 2)]
    )
    mem.commit()
    a = mem.relate()
    b = mem.relate()
    assert a == b  # recompute clears old edges, no duplication


def test_related_unknown_label(mem) -> None:
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    mem.commit()
    mem.relate()
    assert mem.related("ghost") == []
