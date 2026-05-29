"""Migration 001 — initial schema from schema.sql."""

from __future__ import annotations

import sqlite3
from pathlib import Path

VERSION = 1


def up(conn: sqlite3.Connection) -> None:
    sql = (Path(__file__).resolve().parent.parent / "schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
