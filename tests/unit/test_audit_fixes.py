from __future__ import annotations

import json

from tests.conftest import DIM, make_det


def test_label_mass_is_raw_cumulative(mem) -> None:
    # same object seen twice (conf 0.9 each) -> one node, label mass = 1.8 (raw
    # cumulative confidence, not renormalized to <=1)
    mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
    mem.commit()
    mem.add_detections([make_det("mug", (1.02, 0.0, 0.9), 1)])
    mem.commit()
    data = json.loads(mem.serialize(format="json"))
    labels = data["nodes"][0]["labels"]
    assert labels[0][0] == "mug"
    assert labels[0][1] > 1.0  # raw mass accumulates past 1.0 (was clamped before)


def test_relational_anchor_word_boundary(mem) -> None:
    # "able" must not be matched as the anchor for "table" (substring trap)
    mem.add_detections(
        [make_det("table", (1.0, 0.0, 0.4), 1), make_det("able", (5.0, 0.0, 0.4), 2)]
    )
    mem.commit()
    mem.relate()
    res = mem.query("what's on the table")
    assert res.debug.get("anchor") == "table"  # not "able"


def test_budget_drops_empty_parent_region(tmp_path) -> None:
    from tempomem import HashEncoder, SpatialMemory

    enc = HashEncoder(DIM)
    with SpatialMemory.open(tmp_path / "b.smem", embedding_dim=DIM, encoder=enc) as mem:
        mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        mem.commit()
        mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
        # budget that fits only the header -> the whole kitchen subtree is
        # dropped as a unit; no childless "kitchen" line appears
        txt = mem.serialize(format="prompt", max_tokens=8)
        assert "kitchen" not in txt
        assert "more omitted" in txt
