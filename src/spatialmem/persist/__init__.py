"""Persistence: open/create a .smem store, run migrations, expose the connection.

M0 uses BLOB float32 vectors and stdlib sqlite. See spec/SCHEMA.md.
"""

from __future__ import annotations

import os
import sqlite3
from importlib import import_module
from pathlib import Path

from .._errors import SchemaMismatchError, StoreError

SCHEMA_VERSION = 1
CREATOR_VERSION = "0.1.0a1"


def connect(
    path: str | os.PathLike[str], *, embedding_dim: int, readonly: bool, create: bool
) -> sqlite3.Connection:
    p = Path(path)
    if not create and not p.exists():
        raise StoreError(f"store does not exist: {p}")

    uri = f"file:{p}"
    if readonly:
        uri += "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if readonly:
        _check_dim(conn, embedding_dim)
    else:
        _ensure_schema(conn, embedding_dim)

    from .. import vec

    vec.try_enable(conn, embedding_dim, readonly)
    return conn


def _has_meta(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
        is not None
    )


def _applied_version(conn: sqlite3.Connection) -> int:
    if not _has_meta(conn):
        return 0
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    return int(row["value"]) if row else 0


def _ensure_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    current = _applied_version(conn)
    if current == 0:
        with conn:
            mod = import_module("spatialmem.persist.migrations.001_init")
            mod.up(conn)
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("embedding_dim", str(embedding_dim)),
                    ("creator_version", CREATOR_VERSION),
                ],
            )
        return
    if current > SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"store schema v{current} newer than library v{SCHEMA_VERSION}; upgrade spatialmem"
        )
    _check_dim(conn, embedding_dim)


def _check_dim(conn: sqlite3.Connection, embedding_dim: int) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
    if row is None:
        raise SchemaMismatchError("store missing embedding_dim in meta")
    stored = int(row["value"])
    if stored != embedding_dim:
        raise SchemaMismatchError(f"store embedding_dim={stored} but requested {embedding_dim}")
