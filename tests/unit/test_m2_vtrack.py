from __future__ import annotations

import pytest

from tempomem import QueryError, SpatialMemory
from tempomem.bench import recall_at_k
from tests.conftest import DIM, make_det

DAY = 86400.0


# ---- V2 decay ------------------------------------------------------------


def test_decay_lowers_confidence(mem) -> None:
    mem.add_detections([make_det("mug", (0, 0, 0), 1, ts=0.0)])
    mem.commit()
    before = mem.recent(n=1)[0].confidence
    decayed, pruned = mem.decay(half_life_days=10.0, min_conf=0.0, now=10 * DAY)
    assert decayed == 1
    assert pruned == 0
    after = mem.recent(n=1)[0].confidence
    assert after == pytest.approx(before * 0.5, rel=1e-3)


def test_decay_prunes_below_floor(mem) -> None:
    mem.add_detections([make_det("mug", (0, 0, 0), 1, ts=0.0)])
    mem.commit()
    # 100 half-lives -> conf ~0 -> pruned
    _, pruned = mem.decay(half_life_days=1.0, min_conf=0.1, now=100 * DAY)
    assert pruned == 1
    assert mem.stats().n_nodes == 0


def test_decay_no_op_for_fresh_nodes(mem) -> None:
    mem.add_detections([make_det("mug", (0, 0, 0), 1, ts=100.0)])
    mem.commit()
    decayed, pruned = mem.decay(half_life_days=30.0, min_conf=0.1, now=100.0)
    assert decayed == 0 and pruned == 0


def test_decay_bad_half_life(mem) -> None:
    with pytest.raises(ValueError):
        mem.decay(half_life_days=0.0)


# ---- V3 verbalizer -------------------------------------------------------


class StubVerbalizer:
    """Captures the prompt and returns a fixed answer."""

    def __init__(self) -> None:
        self.last_prompt = ""

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "the mug is on the counter"


def test_answer_uses_verbalizer(mem) -> None:
    mem.add_detections([make_det("mug", (1, 0, 1), 1), make_det("sink", (0, 1, 0), 2)])
    mem.commit()
    vb = StubVerbalizer()
    out = mem.answer("where is the mug?", verbalizer=vb)
    assert out == "the mug is on the counter"
    assert "QUESTION: where is the mug?" in vb.last_prompt
    assert "SCENE:" in vb.last_prompt
    assert "mug" in vb.last_prompt


def test_answer_without_verbalizer_raises(mem) -> None:
    mem.add_detections([make_det("mug", (0, 0, 0), 1)])
    mem.commit()
    with pytest.raises(QueryError):
        mem.answer("where is the mug?")


def test_answer_verbalizer_at_open(tmp_path) -> None:
    vb = StubVerbalizer()
    with SpatialMemory.open(tmp_path / "v.smem", embedding_dim=DIM, verbalizer=vb) as mem:
        mem.add_detections([make_det("mug", (0, 0, 0), 1)])
        mem.commit()
        assert mem.answer("x?") == "the mug is on the counter"


# ---- V5 eval harness -----------------------------------------------------


def test_recall_at_k(mem) -> None:
    mem.add_detections(
        [
            make_det("coffee mug", (0, 0, 0), 1),
            make_det("kettle", (5, 0, 0), 2),
            make_det("fridge", (10, 0, 0), 3),
        ]
    )
    mem.commit()
    cases = [("mug", "mug"), ("kettle", "kettle"), ("teapot", "teapot")]
    rep = recall_at_k(mem, cases, k=5)
    assert rep.total == 3
    assert rep.hits == 2  # mug + kettle hit, teapot misses
    assert rep.recall == pytest.approx(2 / 3)
    assert rep.misses == ["teapot"]
