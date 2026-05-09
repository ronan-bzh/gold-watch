"""Unit tests for mine_clusterer module."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from goldmine_watch.data.mine_clusterer import (
    cluster_mines_by_tile,
    get_tile_bbox,
    load_mines,
)

if TYPE_CHECKING:
    pass

TILE_ID_RE = re.compile(r"^T\d{2}[A-Z]{3}$")


@pytest.fixture
def sample_geojson(tmp_path: Path) -> Path:
    """Create a tiny GeoJSON with 3 mines across 2 tiles."""
    geojson_path = tmp_path / "mines.geojson"
    polygons = [
        Polygon([(-53.1, 4.1), (-53.0, 4.1), (-53.0, 4.2), (-53.1, 4.2)]),
        Polygon([(-53.2, 4.3), (-53.1, 4.3), (-53.1, 4.4), (-53.2, 4.4)]),
        Polygon([(-53.6, 4.4), (-53.5, 4.4), (-53.5, 4.5), (-53.6, 4.5)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=polygons,
        crs="EPSG:4326",
    )
    gdf.to_file(geojson_path, driver="GeoJSON")
    return geojson_path


@pytest.fixture
def sample_gdf() -> gpd.GeoDataFrame:
    """Return an in-memory GeoDataFrame with mines spanning 2 tiles."""
    polygons = [
        Polygon([(-53.1, 4.1), (-53.0, 4.1), (-53.0, 4.2), (-53.1, 4.2)]),
        Polygon([(-53.2, 4.3), (-53.1, 4.3), (-53.1, 4.4), (-53.2, 4.4)]),
        Polygon([(-53.6, 4.4), (-53.5, 4.4), (-53.5, 4.5), (-53.6, 4.5)]),
    ]
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=polygons,
        crs="EPSG:4326",
    )


class TestLoadMines:
    """Tests for load_mines."""

    def test_loads_all_mines(self, sample_geojson: Path) -> None:
        """Should return 3 polygons from the fixture."""
        gdf = load_mines(str(sample_geojson))
        assert len(gdf) == 3

    def test_returns_geodataframe(self, sample_geojson: Path) -> None:
        """Should return a GeoDataFrame with geometry column."""
        gdf = load_mines(str(sample_geojson))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "geometry" in gdf.columns


class TestClusterMinesByTile:
    """Tests for cluster_mines_by_tile."""

    def test_clusters_non_empty(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Should return at least one cluster."""
        clusters = cluster_mines_by_tile(sample_gdf)
        assert len(clusters) >= 1

    def test_all_mines_assigned(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Sum of mines across clusters should equal total."""
        clusters = cluster_mines_by_tile(sample_gdf)
        total = sum(len(c) for c in clusters.values())
        assert total == len(sample_gdf)

    def test_known_tile_ids(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Should produce valid Sentinel-2 tile IDs (T21Nxx, T22Nxx)."""
        clusters = cluster_mines_by_tile(sample_gdf)
        for tile_id in clusters:
            assert TILE_ID_RE.match(tile_id), f"Invalid tile ID: {tile_id}"


class TestGetTileBBox:
    """Tests for get_tile_bbox."""

    def test_returns_four_floats(self) -> None:
        """Should return (min_lon, min_lat, max_lon, max_lat)."""
        bbox = get_tile_bbox("T21NZF")
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)

    def test_bbox_valid_for_tile(self) -> None:
        """Bbox should cover the correct UTM zone grid square (~100 km)."""
        bbox = get_tile_bbox("T21NZF")
        min_lon, min_lat, max_lon, max_lat = bbox
        assert min_lon < max_lon
        assert min_lat < max_lat
        lon_span = max_lon - min_lon
        lat_span = max_lat - min_lat
        # Sentinel-2 MGRS tiles are ~100 km; spans vary with latitude.
        assert 0.5 <= lon_span <= 1.5
        assert 0.5 <= lat_span <= 1.5
