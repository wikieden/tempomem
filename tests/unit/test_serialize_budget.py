from __future__ import annotations

from tests.conftest import make_det


def _many(mem, n: int) -> None:
    # n distinct objects, increasing ts so recency order is well-defined
    dets = [make_det(f"obj{i}", (float(i) * 2, 0.0, 0.0), i + 1, ts=float(i)) for i in range(n)]
    mem.add_detections(dets)
    mem.commit()


def test_budget_trims_and_marks_omission(mem) -> None:
    _many(mem, 20)
    full = mem.serialize(format="prompt")
    capped = mem.serialize(format="prompt", max_tokens=40)
    assert len(capped) < len(full)
    assert "more omitted" in capped
    # token estimate (~4 chars/token) stays within budget + the marker line
    body = "\n".join(ln for ln in capped.splitlines() if "more omitted" not in ln)
    assert len(body) // 4 <= 40 + 5


def test_budget_keeps_most_recent_first(mem) -> None:
    _many(mem, 20)
    capped = mem.serialize(format="prompt", max_tokens=40)
    # newest object (obj19, ts=19) must survive; an old one (obj0) should not
    assert '"obj19"' in capped
    assert '"obj0"' not in capped


def test_no_budget_returns_everything(mem) -> None:
    _many(mem, 5)
    txt = mem.serialize(format="prompt")
    assert "more omitted" not in txt
    for i in range(5):
        assert f'"obj{i}"' in txt


def test_node_ids_restricts_to_subgraph(mem) -> None:
    _many(mem, 6)
    target = mem.query("obj3").nodes[0]
    txt = mem.serialize(format="prompt", node_ids={target.id})
    assert '"obj3"' in txt  # the queried node is present
    assert '"obj0"' not in txt  # unrelated nodes are excluded
    assert '"obj5"' not in txt
