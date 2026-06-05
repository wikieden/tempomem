from __future__ import annotations

from spatialmem import HashEncoder, SpatialMemory
from tests.conftest import DIM, make_det


def _seed(mem) -> None:
    # kitchen cluster around x~1, a lone object far away at x~9
    mem.add_detections(
        [
            make_det("mug", (1.0, 0.0, 0.9), 1),
            make_det("kettle", (1.4, 0.0, 0.9), 2),
            make_det("sink", (0.6, 0.0, 0.9), 3),
            make_det("doormat", (9.0, 0.0, 0.0), 4),
        ]
    )
    mem.commit()


def test_define_region_adopts_inside_objects(mem) -> None:
    _seed(mem)
    rid = mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    kids = mem.contents(rid)
    labels = sorted(h.label for h in kids)
    assert labels == ["kettle", "mug", "sink"]  # doormat at x=9 excluded
    assert mem.stats().n_nodes == 5  # 4 objects + 1 room


def test_contents_by_label(mem) -> None:
    _seed(mem)
    mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    assert sorted(h.label for h in mem.contents("kitchen")) == ["kettle", "mug", "sink"]
    assert mem.contents("nonexistent") == []


def test_region_does_not_swallow_observations(mem) -> None:
    _seed(mem)
    mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    # a new mug sighting inside the room must merge into the mug object, not the room
    mem.add_detections([make_det("mug", (1.02, 0.0, 0.9), 1)])
    mem.commit()
    assert mem.stats().n_nodes == 5  # unchanged: no new node, no merge-into-room


def test_region_queryable_with_encoder(tmp_path) -> None:
    enc = HashEncoder(DIM)
    with SpatialMemory.open(tmp_path / "h.smem", embedding_dim=DIM, encoder=enc) as mem:
        mem.add_detections([make_det("mug", (1.0, 0.0, 0.9), 1)])
        mem.commit()
        mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
        res = mem.query("kitchen")
        assert res.nodes and res.nodes[0].label == "kitchen"


def test_forget_region_with_children_no_crash(mem) -> None:
    _seed(mem)
    rid = mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    mem.forget(rid)  # must not raise FK IntegrityError
    assert mem.stats().n_nodes == 4  # room gone, 4 objects survive
    # children reparented to top level
    assert mem.serialize(format="prompt").count('"mug"') == 1


def test_decay_does_not_prune_region(mem) -> None:
    _seed(mem)
    mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    # huge age would prune objects, but the room must survive (and not crash)
    mem.decay(half_life_days=1.0, min_conf=0.5, now=1000.0 + 100 * 86400)
    assert sorted(h.label for h in mem.contents("kitchen")) == []  # objects pruned, room kept
    assert any(n.label == "kitchen" for n in mem.recent(n=10))


def test_define_region_idempotent(mem) -> None:
    _seed(mem)
    r1 = mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    r2 = mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    assert r1 == r2  # reused, not duplicated
    assert sum(1 for n in mem.recent(n=20) if n.label == "kitchen") == 1
    assert sorted(h.label for h in mem.contents("kitchen")) == ["kettle", "mug", "sink"]


def test_serialize_prompt_nests_children(mem) -> None:
    _seed(mem)
    mem.define_region("kitchen", (0.0, -1.0, 0.0), (2.0, 1.0, 2.0))
    txt = mem.serialize(format="prompt")
    lines = txt.splitlines()
    room_line = next(i for i, ln in enumerate(lines) if "kitchen" in ln)
    child = lines[room_line + 1]
    assert child.startswith("    ")  # deeper indent than the room's two spaces
