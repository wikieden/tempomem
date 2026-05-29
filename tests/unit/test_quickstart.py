from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "01_quickstart.py"


def test_quickstart_runs() -> None:
    spec = importlib.util.spec_from_file_location("qs_demo", EXAMPLE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()  # asserts internally; must not raise
