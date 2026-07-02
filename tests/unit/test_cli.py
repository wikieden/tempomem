from __future__ import annotations

import pytest

from tempomem import TempoMem
from tempomem.cli import main
from tests.conftest import DIM, make_det


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "tempomem" in capsys.readouterr().out


def test_inspect(tmp_path, capsys) -> None:
    p = tmp_path / "c.smem"
    with TempoMem.open(p, embedding_dim=DIM) as m:
        m.add_detections([make_det("mug", (1, 0, 0), 1)])
        m.commit()
    rc = main(["inspect", str(p), "--embedding-dim", str(DIM)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nodes:      1" in out
    assert "mug" in out


def test_no_command_returns_1(capsys) -> None:
    assert main([]) == 1
