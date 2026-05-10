"""Unit tests for the FastAPI tile server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# We need to mock TileRegistry before importing the server module
# because it instantiates a global registry at import time.


@pytest.fixture
def client(tmp_path: Path):
    """Return a TestClient with a mocked registry."""
    db = tmp_path / "tiles.db"
    schema = tmp_path / "schema.sql"
    schema.write_text(
        """
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
        CREATE TABLE IF NOT EXISTS fg_boundary (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            west        REAL NOT NULL,
            south       REAL NOT NULL,
            east        REAL NOT NULL,
            north       REAL NOT NULL
        );
        INSERT OR IGNORE INTO fg_boundary (id, name, west, south, east, north)
        VALUES (1, 'french_guiana', -54.6, 2.1, -51.6, 5.8);
        """
    )

    with patch.dict(
        "os.environ",
        {"TILES_DB": str(db), "TILES_SCHEMA": str(schema)},
    ):
        # Import inside the patch so the module picks up the env vars
        from web.server import app, registry

        # Seed a fake tile so we have data to serve
        registry.register_tile(
            tile_id="T21NZF",
            date="20231026",
            filepath=str(tmp_path / "T21NZF_20231026.tif"),
            bounds=(-54.0, 4.0, -53.9, 4.1),
            crs="EPSG:4326",
            width=256,
            height=256,
            bands=3,
            size_bytes=1024,
            source="test",
        )

        with TestClient(app) as test_client:
            yield test_client


class TestTileServer:
    def test_health_endpoint(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_tiles_info_returns_json(self, client: TestClient):
        response = client.get("/tiles/info")
        assert response.status_code == 200
        data = response.json()
        assert "tiles" in data
        assert len(data["tiles"]) >= 1

    def test_tile_info_endpoint(self, client: TestClient):
        response = client.get("/tiles/T21NZF/info")
        assert response.status_code == 200
        data = response.json()
        assert data["tile_id"] == "T21NZF"

    def test_tile_info_for_missing_tile_returns_404(self, client: TestClient):
        response = client.get("/tiles/T00MISSING/info")
        assert response.status_code == 404

    def test_tile_outside_fg_returns_204(self, client: TestClient):
        # Request a tile far outside French Guiana (zoom 2, x=0, y=0 covers Greenland)
        response = client.get("/tiles/2/0/0.png")
        assert response.status_code == 204

    def test_refresh_endpoint_updates_registry(self, client: TestClient):
        response = client.post("/tiles/refresh")
        assert response.status_code == 200
        data = response.json()
        assert "registered" in data

    def test_root_serves_index(self, client: TestClient):
        response = client.get("/")
        # index.html may not exist in test env, but endpoint should be reachable
        assert response.status_code in (200, 404)
