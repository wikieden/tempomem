from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "02_query_and_answer.py"


def test_example02_runs() -> None:
    spec = importlib.util.spec_from_file_location("ex02", EXAMPLE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()  # must run end to end without raising
