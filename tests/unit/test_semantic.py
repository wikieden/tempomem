from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tempomem import Detection, SchemaMismatchError, TempoMem
from tempomem._errors import StoreError
from tests.conftest import make_det

DIM = 16


def _seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


class StubEncoder:
    """Deterministic text encoder: string -> seeded unit vector.

    Stands in for a real CLIP encoder so semantic retrieval is testable
    without Torch. Same string always yields the same vector.
    """

    @property
    def dim(self) -> int:
        return DIM

    def encode_text(self, texts) -> np.ndarray:
        rows = []
        for t in texts:
            rng = np.random.default_rng(_seed(t))
            v = rng.standard_normal(DIM).astype("float32")
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype="float32")


def _det(enc: StubEncoder, label: str, center) -> Detection:
    cx, cy, cz = center
    return Detection(
        label=label,
        feature=enc.encode_text([label])[0],
        center_xyz=center,
        bbox_min=(cx - 0.05, cy - 0.05, cz - 0.05),
        bbox_max=(cx + 0.05, cy + 0.05, cz + 0.05),
        confidence=0.9,
        ts=1.0,
    )


def test_semantic_query_ranks_by_embedding(tmp_path) -> None:
    enc = StubEncoder()
    with TempoMem.open(tmp_path / "s.smem", embedding_dim=DIM, encoder=enc) as mem:
        mem.add_detections(
            [
                _det(enc, "mug", (0, 0, 0)),
                _det(enc, "kettle", (5, 0, 0)),
                _det(enc, "fridge", (10, 0, 0)),
            ]
        )
        mem.commit()
        res = mem.query("mug")
        assert res.debug.get("encoder") is True
        assert res.nodes[0].label == "mug"
        assert res.nodes[0].score > 0.99  # exact embedding match


def test_semantic_method_uses_encoder(tmp_path) -> None:
    enc = StubEncoder()
    with TempoMem.open(tmp_path / "s2.smem", embedding_dim=DIM, encoder=enc) as mem:
        mem.add_detections([_det(enc, "kettle", (0, 0, 0)), _det(enc, "mug", (5, 0, 0))])
        mem.commit()
        hits = mem.semantic("kettle")
        assert hits[0].label == "kettle"


def test_semantic_without_encoder_falls_back_to_keyword(tmp_path) -> None:
    enc = StubEncoder()
    with TempoMem.open(tmp_path / "s3.smem", embedding_dim=DIM) as mem:  # no encoder
        mem.add_detections([_det(enc, "coffee mug", (0, 0, 0))])
        mem.commit()
        hits = mem.semantic("mug")
        assert hits and hits[0].label == "coffee mug"


def test_encoder_dim_mismatch_rejected(tmp_path) -> None:
    enc = StubEncoder()  # dim=16
    with pytest.raises(SchemaMismatchError):
        TempoMem.open(tmp_path / "s4.smem", embedding_dim=32, encoder=enc)


# ---------------------------------------------------------------------------
# D-B: semantic edges, node properties, event timeline
# ---------------------------------------------------------------------------


def test_add_edge_by_id(mem) -> None:
    from tempomem import store as _store

    mem.add_detections([make_det("guest", (0, 0, 0), 1), make_det("coffee", (1, 0, 0), 2)])
    mem.commit()
    all_nodes = _store.all_nodes(mem._conn)
    guest = next(n for n in all_nodes if n.label == "guest")
    coffee = next(n for n in all_nodes if n.label == "coffee")

    mem.add_edge(guest.id, "prefers", coffee.id, ts=1.0)

    edges = mem.get_edges(guest.id)
    assert len(edges) == 1
    assert edges[0][0].label == "coffee"
    assert edges[0][1] == "prefers"


def test_add_edge_by_label(mem) -> None:
    mem.add_detections([make_det("guest", (0, 0, 0), 1), make_det("tea", (1, 0, 0), 2)])
    mem.commit()

    mem.add_edge("guest", "prefers", "tea")

    edges = mem.get_edges("guest", rel="prefers")
    assert edges[0][0].label == "tea"


def test_add_edge_idempotent(mem) -> None:
    mem.add_detections([make_det("a", (0, 0, 0), 1), make_det("b", (1, 0, 0), 2)])
    mem.commit()

    mem.add_edge("a", "knows", "b", ts=1.0)
    mem.add_edge("a", "knows", "b", ts=2.0)  # idempotent, updates ts

    assert len(mem.get_edges("a", rel="knows")) == 1


def test_get_edges_direction_in(mem) -> None:
    mem.add_detections([make_det("guest", (0, 0, 0), 1), make_det("milk", (1, 0, 0), 2)])
    mem.commit()

    mem.add_edge("guest", "allergic_to", "milk")

    incoming = mem.get_edges("milk", rel="allergic_to", direction="in")
    assert incoming[0][0].label == "guest"
    assert incoming[0][1] == "allergic_to"


def test_add_edge_unknown_label_raises(mem) -> None:
    with pytest.raises(StoreError):
        mem.add_edge("ghost", "knows", "nobody")


def test_set_and_get_property(mem) -> None:
    mem.add_detections([make_det("guest", (0, 0, 0), 1)])
    mem.commit()

    mem.set_property("guest", "vip", True, ts=1.0)
    mem.set_property("guest", "drink_count", 3)

    assert mem.get_property("guest", "vip") is True
    assert mem.get_property("guest", "drink_count") == 3


def test_set_property_overwrites(mem) -> None:
    mem.add_detections([make_det("guest", (0, 0, 0), 1)])
    mem.commit()

    mem.set_property("guest", "mood", "happy")
    mem.set_property("guest", "mood", "tired")

    assert mem.get_property("guest", "mood") == "tired"


def test_get_property_missing_returns_none(mem) -> None:
    mem.add_detections([make_det("guest", (0, 0, 0), 1)])
    mem.commit()

    assert mem.get_property("guest", "nonexistent") is None


def test_get_property_unknown_node_returns_none(mem) -> None:
    assert mem.get_property(9999, "key") is None


def test_add_event_and_query(mem) -> None:
    mem.add_detections([make_det("lobby", (0, 0, 0), 1)])
    mem.commit()

    eid = mem.add_event("guest_arrived", location="lobby", ts=100.0, payload={"name": "Alice"})
    assert isinstance(eid, int)

    events = mem.query_events("guest_arrived")
    assert len(events) == 1
    assert events[0]["payload"]["name"] == "Alice"
    assert events[0]["ts"] == 100.0


def test_query_events_since_ts(mem) -> None:
    mem.add_detections([make_det("lobby", (0, 0, 0), 1)])
    mem.commit()

    mem.add_event("guest_arrived", ts=50.0)
    mem.add_event("guest_arrived", ts=150.0)
    mem.add_event("guest_arrived", ts=250.0)

    recent = mem.query_events("guest_arrived", since_ts=100.0)
    assert len(recent) == 2
    assert all(e["ts"] >= 100.0 for e in recent)


def test_query_events_by_region(mem) -> None:
    mem.add_detections([make_det("lobby", (0, 0, 0), 1), make_det("kitchen", (5, 0, 0), 2)])
    mem.commit()

    mem.add_event("motion", location="lobby", ts=1.0)
    mem.add_event("motion", location="kitchen", ts=2.0)

    lobby_events = mem.query_events("motion", region="lobby")
    assert len(lobby_events) == 1
    assert lobby_events[0]["ts"] == 1.0


def test_last_changed_by_node_update(mem) -> None:
    # Add node first so define_region assigns it as a child.
    mem.add_detections([make_det("chair", (0, 0, 0), 1, ts=500.0)])
    mem.commit()
    lobby = mem.define_region("lobby", bbox_min=(-5, -5, -5), bbox_max=(5, 5, 5))

    ts, change_type = mem.last_changed(lobby)
    assert ts is not None
    assert change_type == "node_update"


def test_last_changed_by_event(mem) -> None:
    lobby = mem.define_region("lobby", bbox_min=(-5, -5, -5), bbox_max=(5, 5, 5))

    mem.add_event("motion", location=lobby, ts=999.0)

    ts, change_type = mem.last_changed(lobby)
    assert ts == 999.0
    assert change_type == "motion"


def test_last_changed_unknown_region_returns_none(mem) -> None:
    ts, change_type = mem.last_changed(9999)
    assert ts is None
    assert change_type is None


def test_schema_v1_store_upgrades_to_v2(tmp_path) -> None:
    """A store created against library v1 must be transparently upgraded to v2."""
    import sqlite3

    from tempomem.persist import SCHEMA_VERSION

    # Simulate a v1 store: run only 001_init, set schema_version=1.
    path = tmp_path / "old.smem"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    from importlib import import_module
    mod = import_module("tempomem.persist.migrations.001_init")
    with conn:
        mod.up(conn)
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?)",
            [("schema_version", "1"), ("embedding_dim", "16"), ("creator_version", "0.1.0a0")],
        )
    conn.close()

    # Re-open with current library: should upgrade silently.
    with TempoMem.open(path, embedding_dim=16) as mem:
        ver = int(
            mem._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[
                "value"
            ]
        )
        assert ver == SCHEMA_VERSION
        # Verify new tables exist.
        tables = {
            r[0]
            for r in mem._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "semantic_edges" in tables
        assert "node_properties" in tables
        assert "smem_events" in tables
