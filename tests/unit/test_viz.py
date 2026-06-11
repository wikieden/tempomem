from __future__ import annotations

from tempomem import SpatialMemory, viz
from tempomem.cli import main
from tests.conftest import DIM, make_det


def test_to_html_contains_nodes(mem) -> None:
    mem.add_detections([make_det("mug", (1, 0, 0), 1), make_det("sink", (0, 1, 0), 2)])
    mem.commit()
    html = viz.to_html(mem._conn, DIM, title="kitchen")
    assert html.startswith("<!doctype html>")
    assert "2 nodes" in html
    assert "mug" in html and "sink" in html
    assert "kitchen" in html


def test_to_html_empty_store(mem) -> None:
    html = viz.to_html(mem._conn, DIM)
    assert "0 nodes" in html
    assert "<!doctype html>" in html


def test_cli_viz_writes_file(tmp_path, capsys) -> None:
    p = tmp_path / "v.smem"
    with SpatialMemory.open(p, embedding_dim=DIM) as m:
        m.add_detections([make_det("mug", (1, 0, 0), 1)])
        m.commit()
    out = tmp_path / "scene.html"
    rc = main(["viz", str(p), "-o", str(out), "--embedding-dim", str(DIM)])
    assert rc == 0
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "mug" in body and "1 nodes" in body
    assert "wrote" in capsys.readouterr().out
