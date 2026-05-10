-- GoldMine Watch — Tile Registry Schema
-- This file is the single source of truth for the SQLite database structure.

CREATE TABLE IF NOT EXISTS tiles (
    id          INTEGER PRIMARY KEY,
    tile_id     TEXT NOT NULL,
    date        TEXT NOT NULL,
    filepath    TEXT NOT NULL UNIQUE,
    west        REAL NOT NULL,
    south       REAL NOT NULL,
    east        REAL NOT NULL,
    north       REAL NOT NULL,
    crs         TEXT NOT NULL DEFAULT 'EPSG:4326',
    width       INTEGER,
    height      INTEGER,
    bands       INTEGER,
    size_bytes  INTEGER,
    source      TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(tile_id, date)
);

CREATE INDEX IF NOT EXISTS idx_tiles_tile_id ON tiles(tile_id);
CREATE INDEX IF NOT EXISTS idx_tiles_bounds ON tiles(west, south, east, north);

CREATE TABLE IF NOT EXISTS fg_boundary (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    west        REAL NOT NULL,
    south       REAL NOT NULL,
    east        REAL NOT NULL,
    north       REAL NOT NULL
);

-- Seed French Guiana boundary on first init
INSERT OR IGNORE INTO fg_boundary (id, name, west, south, east, north)
VALUES (1, 'french_guiana', -54.6, 2.1, -51.6, 5.8);
