"""Tests for the TileCache module."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from requests_mock import Mocker

from goldmine_watch.data.tile_cache import TileCache, get_tile_id_from_bbox

# Saint-Laurent-du-Maroni bounding box
SAINT_LAURENT_BBOX = (-54.05, 5.45, -54.0, 5.49)


def _make_geotiff_bytes(
    width: int = 64, height: int = 64, count: int = 7, dtype: str = "uint16"
) -> bytes:
    """Create a small GeoTIFF in memory and return its bytes."""
    transform = from_origin(0, 0, 10, 10)
    data = np.random.randint(0, 1000, size=(count, height, width), dtype=dtype)
    with MemoryFile() as memfile:
        with rasterio.open(
            memfile,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=data.dtype,
            crs="EPSG:32622",
            transform=transform,
        ) as dst:
            dst.write(data)
        return bytes(memfile.getbuffer())


def _write_geotiff(path: Path, count: int = 7) -> None:
    """Write a small valid GeoTIFF to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0, 0, 10, 10)
    data = np.random.randint(0, 1000, size=(count, 64, 64), dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=count,
        dtype=data.dtype,
        crs="EPSG:32622",
        transform=transform,
    ) as dst:
        dst.write(data)


class TestTileCache:
    """Unit tests for TileCache."""

    def test_cache_dir_created(self, tmp_path: Path) -> None:
        """Should create cache directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()
        TileCache(str(cache_dir))
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_tile_not_in_cache_triggers_download(
        self, tmp_path: Path, requests_mock: Mocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should download tile when not found in cache."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")

        requests_mock.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            json={
                "access_token": "mock_token",
                "token_type": "Bearer",
                "expires_in": 600,
            },
        )
        requests_mock.post(
            "https://catalogue.dataspace.copernicus.eu/stac/search",
            json={
                "features": [
                    {
                        "id": "S2A_T21NZG_20231026T",
                        "bbox": list(SAINT_LAURENT_BBOX),
                        "properties": {
                            "datetime": "2023-10-26T10:30:00Z",
                            "eo:cloud_cover": 5.0,
                        },
                    }
                ]
            },
        )
        requests_mock.post(
            "https://sh.dataspace.copernicus.eu/api/v1/process",
            content=_make_geotiff_bytes(count=7),
        )

        cache = TileCache(str(tmp_path / "cache"))
        result = cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )

        assert result.exists()
        assert result.name == "T21NZG_20231026.tif"
        with rasterio.open(result) as src:
            assert src.count == 7

    def test_tile_in_cache_returns_immediately(self, tmp_path: Path, requests_mock: Mocker) -> None:
        """Should return cached tile without downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / "T21NZG_20231026.tif"
        _write_geotiff(cached_path)

        cache = TileCache(str(cache_dir))
        result = cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )

        assert result == cached_path
        # No HTTP requests should have been made
        assert len(requests_mock.request_history) == 0

    def test_list_cached_tiles_returns_paths(self, tmp_path: Path) -> None:
        """Should return list of cached tile paths."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_geotiff(cache_dir / "T21NZG_20231026.tif")
        _write_geotiff(cache_dir / "T21NZG_20231115.tif")

        cache = TileCache(str(cache_dir))
        tiles = cache.list_cached_tiles()

        assert len(tiles) == 2
        assert all(p.suffix == ".tif" for p in tiles)

    def test_cache_size_computed_correctly(self, tmp_path: Path) -> None:
        """Should sum sizes of all cached files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path1 = cache_dir / "tile1.tif"
        path2 = cache_dir / "tile2.tif"
        _write_geotiff(path1)
        _write_geotiff(path2)

        cache = TileCache(str(cache_dir))
        size_mb = cache.get_cache_size_mb()

        expected_bytes = path1.stat().st_size + path2.stat().st_size
        expected_mb = expected_bytes / (1024 * 1024)
        assert size_mb == pytest.approx(expected_mb, rel=1e-6)

    def test_clear_cache_removes_files(self, tmp_path: Path) -> None:
        """Should delete all cached tiles."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_geotiff(cache_dir / "tile1.tif")
        _write_geotiff(cache_dir / "tile2.tif")

        cache = TileCache(str(cache_dir))
        assert len(cache.list_cached_tiles()) == 2

        cache.clear_cache()
        assert len(cache.list_cached_tiles()) == 0


class TestGetTileIdFromBbox:
    """Tests for get_tile_id_from_bbox utility."""

    def test_known_bbox_returns_correct_tile(self) -> None:
        """Saint-Laurent-du-Maroni bbox should return T21NZG."""
        tile_id = get_tile_id_from_bbox(SAINT_LAURENT_BBOX)
        assert tile_id == "T21NZG"

    def test_invalid_bbox_raises(self) -> None:
        """Bbox outside Sentinel-2 coverage should raise ValueError."""
        with pytest.raises(ValueError):
            get_tile_id_from_bbox((-10.0, 85.0, -9.0, 86.0))

        with pytest.raises(ValueError):
            get_tile_id_from_bbox((-10.0, -85.0, -9.0, -81.0))
