-- Chronotope store schema. See spec/SCHEMA.md.
-- M0 note: feature vectors stored as BLOB (float32). sqlite-vec ANN tables
-- arrive in M1; logical shape (obs_features / node_features) is preserved.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id       INTEGER PRIMARY KEY,
    session  TEXT NOT NULL DEFAULT 'default',
    label    TEXT,
    start_ts REAL,
    end_ts   REAL
);

CREATE TABLE IF NOT EXISTS observations (
    id         INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    ts         REAL NOT NULL,
    label      TEXT NOT NULL,
    confidence REAL NOT NULL,
    center_x   REAL NOT NULL, center_y REAL NOT NULL, center_z REAL NOT NULL,
    bbox_min_x REAL NOT NULL, bbox_min_y REAL NOT NULL, bbox_min_z REAL NOT NULL,
    bbox_max_x REAL NOT NULL, bbox_max_y REAL NOT NULL, bbox_max_z REAL NOT NULL,
    feature    BLOB NOT NULL,
    mask_rle   BLOB,
    aux        TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_episode_ts ON observations(episode_id, ts);

CREATE TABLE IF NOT EXISTS nodes (
    id          INTEGER PRIMARY KEY,
    type        TEXT NOT NULL,
    label       TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    confidence  REAL NOT NULL,
    centroid_x  REAL NOT NULL, centroid_y REAL NOT NULL, centroid_z REAL NOT NULL,
    bbox_min_x  REAL NOT NULL, bbox_min_y REAL NOT NULL, bbox_min_z REAL NOT NULL,
    bbox_max_x  REAL NOT NULL, bbox_max_y REAL NOT NULL, bbox_max_z REAL NOT NULL,
    feature     BLOB NOT NULL,
    n_obs       INTEGER NOT NULL,
    t_first     REAL NOT NULL,
    t_last      REAL NOT NULL,
    parent_id   INTEGER REFERENCES nodes(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
CREATE INDEX IF NOT EXISTS idx_nodes_tlast ON nodes(t_last DESC);

CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY,
    src        INTEGER NOT NULL REFERENCES nodes(id),
    dst        INTEGER NOT NULL REFERENCES nodes(id),
    type       TEXT NOT NULL,
    confidence REAL NOT NULL,
    t_last     REAL NOT NULL,
    aux        TEXT,
    UNIQUE(src, dst, type)
);

CREATE TABLE IF NOT EXISTS node_obs (
    node_id INTEGER NOT NULL REFERENCES nodes(id),
    obs_id  INTEGER NOT NULL REFERENCES observations(id),
    ts      REAL NOT NULL,
    PRIMARY KEY (node_id, obs_id)
);
