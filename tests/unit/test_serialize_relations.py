from __future__ import annotations

from tests.conftest import make_det


def test_prompt_includes_relations_after_relate(mem) -> None:
    mem.add_detections(
        [make_det("table", (1.0, 0.0, 0.40), 1), make_det("book", (1.0, 0.0, 0.55), 2)]
    )
    mem.commit()
    mem.relate()
    txt = mem.serialize(format="prompt")
    # the book line carries its relation suffix
    book_line = next(ln for ln in txt.splitlines() if '"book"' in ln)
    assert "|" in book_line
    assert "on table#" in book_line


def test_prompt_no_relations_before_relate(mem) -> None:
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    mem.commit()
    txt = mem.serialize(format="prompt")
    assert "|" not in txt  # no edges inferred yet


def test_relations_flag_off(mem) -> None:
    mem.add_detections(
        [make_det("table", (1.0, 0.0, 0.40), 1), make_det("book", (1.0, 0.0, 0.55), 2)]
    )
    mem.commit()
    mem.relate()
    txt = mem.serialize(format="prompt", relations=False)
    assert "|" not in txt
