"""Migration 002: semantic edges, node properties, and event timeline.

Adds three tables distinct from the geometric `edges` table:
- semantic_edges: typed directed edges (src --rel--> dst)
- node_properties: key-value attribute bag per node (latest value wins)
- smem_events: append-only semantic event log with optional location ref
"""

from __future__ import annotations

import sqlite3

VERSION = 2


def up(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS semantic_edges (
            id    INTEGER PRIMARY KEY,
            src   INTEGER NOT NULL REFERENCES nodes(id),
            rel   TEXT    NOT NULL,
            dst   INTEGER NOT NULL REFERENCES nodes(id),
            ts    REAL,
            UNIQUE(src, rel, dst)
        );
        CREATE INDEX IF NOT EXISTS idx_sem_edges_src_rel ON semantic_edges(src, rel);
        CREATE INDEX IF NOT EXISTS idx_sem_edges_dst_rel ON semantic_edges(dst, rel);

        CREATE TABLE IF NOT EXISTS node_properties (
            node_id INTEGER NOT NULL REFERENCES nodes(id),
            key     TEXT    NOT NULL,
            value   TEXT    NOT NULL,
            ts      REAL,
            PRIMARY KEY (node_id, key)
        );

        CREATE TABLE IF NOT EXISTS smem_events (
            id       INTEGER PRIMARY KEY,
            type     TEXT    NOT NULL,
            location INTEGER REFERENCES nodes(id),
            ts       REAL    NOT NULL,
            payload  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_smem_events_type_ts ON smem_events(type, ts);
        CREATE INDEX IF NOT EXISTS idx_smem_events_loc_ts  ON smem_events(location, ts);
    """)
