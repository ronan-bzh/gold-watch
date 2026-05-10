"""Unit tests for TileRegistry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from goldmine_watch.data.tile_registry import TileRegistry


@pytest.fixture
def registry(tmp_path: Path) -> TileRegistry:
    """Return a fresh TileRegistry backed by a temp database."""
    db = tmp_path / "tiles.db"
    schema = tmp_path / "schema.sql"
    schema.write_text(
        """
        CREATE TABLE IF NOT EXISTS tiles (
            id INTEGER PRIMARY KEY,
            tile_id TEXT NOT NULL,
            date TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            west REAL NOT NULL,
            south REAL NOT NULL,
            east REAL NOT NULL,
            north REAL NOT NULL,
            crs TEXT NOT NULL DEFAULT 'EPSG:4326',
            width INTEGER,
            height INTEGER,
            bands INTEGER,
            size_bytes INTEGER,
            source TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(tile_id, date)
        );
        CREATE INDEX IF NOT EXISTS idx_tiles_tile_id ON tiles(tile_id);
        CREATE INDEX IF NOT EXISTS idx_tiles_bounds ON tiles(west, south, east, north);
        CREATE TABLE IF NOT EXISTS fg_boundary (
            id INTEGER PRIMARY KEY,
            name TEXT,
            west REAL NOT NULL,
            south REAL NOT NULL,
            east REAL NOT NULL,
            north REAL NOT NULL
        );
        INSERT OR IGNORE INTO fg_boundary (id, name, west, south, east, north)
        VALUES (1, 'french_guiana', -54.6, 2.1, -51.6, 5.8);
        """
    )
    return TileRegistry(db_path=str(db), schema_path=str(schema))


@pytest.fixture
def fake_tile(tmp_path: Path) -> Path:
    """Create a minimal valid GeoTIFF for testing."""
    path = tmp_path / "T21NZF_20231026.tif"
    height, width, bands = 10, 10, 3
    data = np.random.randint(0, 255, (bands, height, width), dtype=np.uint8)
    transform = from_bounds(-54.0, 4.0, -53.9, 4.1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


class TestTileRegistry:
    def test_registry_creates_database_from_sql_file(self, tmp_path: Path):
        db = tmp_path / "tiles.db"
        schema = tmp_path / "schema.sql"
        schema.write_text("CREATE TABLE test (id INTEGER PRIMARY KEY);")
        reg = TileRegistry(db_path=str(db), schema_path=str(schema))
        assert db.exists()

    def test_register_tile_inserts_record(self, registry: TileRegistry):
        row_id = registry.register_tile(
            tile_id="T21NZF",
            date="20231026",
            filepath="/tmp/test.tif",
            bounds=(-54.0, 4.0, -53.9, 4.1),
            width=100,
            height=100,
            bands=3,
        )
        assert row_id > 0
        tile = registry.get_tile("T21NZF")
        assert tile is not None
        assert tile["tile_id"] == "T21NZF"
        assert tile["width"] == 100

    def test_register_tile_rejects_out_of_bounds(self, registry: TileRegistry):
        with pytest.raises(ValueError):
            registry.register_tile(
                tile_id="T00BAD",
                date="20231026",
                filepath="/tmp/test.tif",
                bounds=(0.0, 0.0, 1.0, 1.0),
            )

    def test_get_tile_returns_latest_by_default(self, registry: TileRegistry):
        registry.register_tile(
            tile_id="T21NZF",
            date="20231025",
            filepath="/tmp/old.tif",
            bounds=(-54.0, 4.0, -53.9, 4.1),
        )
        registry.register_tile(
            tile_id="T21NZF",
            date="20231026",
            filepath="/tmp/new.tif",
            bounds=(-54.0, 4.0, -53.9, 4.1),
        )
        tile = registry.get_tile("T21NZF")
        assert tile is not None
        assert tile["date"] == "20231026"

    def test_list_tiles_filters_by_viewport(self, registry: TileRegistry):
        registry.register_tile(
            tile_id="T21NZF",
            date="20231026",
            filepath="/tmp/a.tif",
            bounds=(-54.0, 4.0, -53.9, 4.1),
        )
        registry.register_tile(
            tile_id="T22NBL",
            date="20231026",
            filepath="/tmp/b.tif",
            bounds=(-53.0, 3.0, -52.9, 3.1),
        )
        tiles = registry.list_tiles(intersects_bounds=(-54.5, 3.5, -53.5, 4.5))
        assert len(tiles) == 1
        assert tiles[0]["tile_id"] == "T21NZF"

    def test_refresh_from_disk_finds_new_files(self, registry: TileRegistry, fake_tile: Path):
        count = registry.refresh_from_disk(str(fake_tile.parent))
        assert count == 1
        tile = registry.get_tile("T21NZF", "20231026")
        assert tile is not None
        assert tile["width"] == 10

    def test_delete_tile_removes_record(self, registry: TileRegistry):
        registry.register_tile(
            tile_id="T21NZF",
            date="20231026",
            filepath="/tmp/test.tif",
            bounds=(-54.0, 4.0, -53.9, 4.1),
        )
        registry.delete_tile("T21NZF", "20231026")
        assert registry.get_tile("T21NZF") is None

    def test_fg_boundary_is_seeded_from_sql(self, registry: TileRegistry):
        boundary = registry.get_fg_boundary()
        assert boundary["name"] == "french_guiana"
        assert boundary["west"] == pytest.approx(-54.6)
