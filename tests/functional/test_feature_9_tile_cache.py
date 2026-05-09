"""Functional tests for Feature 9: Unified Tile Cache System.

These tests exercise the full cache workflow end-to-end, including
cache hits, misses, corruption handling, and credential validation.
"""

from pathlib import Path

import pytest
import rasterio
from rasterio.transform import from_origin
from requests_mock import Mocker

from goldmine_watch.data.tile_cache import TileCache

# Saint-Laurent-du-Maroni bounding box
SAINT_LAURENT_BBOX = (-54.05, 5.45, -54.0, 5.49)


def _make_geotiff_bytes(width: int = 64, height: int = 64, count: int = 7) -> bytes:
    """Create a small GeoTIFF in memory and return its bytes."""
    import numpy as np
    from rasterio.io import MemoryFile

    transform = from_origin(0, 0, 10, 10)
    data = np.random.randint(0, 1000, size=(count, height, width), dtype=np.uint16)
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


def _register_mocks(requests_mock: Mocker) -> None:
    """Register mocked Copernicus API endpoints."""
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


class TestFeature9TileCacheFlow:
    """End-to-end tile cache workflow tests."""

    def test_first_download_then_cache_reuse(
        self, tmp_path: Path, requests_mock: Mocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First call downloads, second call returns cached version."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        cache = TileCache(str(tmp_path / "cache"))
        result1 = cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )

        assert result1.exists()
        assert result1.name == "T21NZG_20231026.tif"

        # Second call should return the same path without new downloads
        result2 = cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )
        assert result2 == result1

    def test_cache_avoids_duplicate_downloads(
        self, tmp_path: Path, requests_mock: Mocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple requests for same tile should only download once."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        cache = TileCache(str(tmp_path / "cache"))
        cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )
        cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )
        cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )

        # Count how many times the Process API was called
        process_calls = [
            r
            for r in requests_mock.request_history
            if r.url == "https://sh.dataspace.copernicus.eu/api/v1/process"
        ]
        assert len(process_calls) == 1

    def test_different_tiles_use_different_cache_keys(
        self, tmp_path: Path, requests_mock: Mocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tiles with different IDs should not collide in cache."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        cache = TileCache(str(tmp_path / "cache"))
        result_a = cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )
        result_b = cache.get_tile(
            tile_id="T22NAM",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )

        assert result_a.name == "T21NZG_20231026.tif"
        assert result_b.name == "T22NAM_20231026.tif"
        assert result_a != result_b
        assert result_a.exists()
        assert result_b.exists()

    def test_corrupted_cache_file_re_downloads(
        self, tmp_path: Path, requests_mock: Mocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid/corrupted cached file should trigger re-download."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        corrupted = cache_dir / "T21NZG_20231026.tif"
        corrupted.write_text("this is not a geotiff")

        cache = TileCache(str(cache_dir))
        result = cache.get_tile(
            tile_id="T21NZG",
            date_range="2023-10-01/2023-10-31",
            bbox=SAINT_LAURENT_BBOX,
        )

        assert result.exists()
        assert result.name == "T21NZG_20231026.tif"
        # Verify the corrupted file was replaced with a valid GeoTIFF
        with rasterio.open(result) as src:
            assert src.count == 7

    def test_env_credentials_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing COPERNICUS_CLIENT_ID should raise error."""
        monkeypatch.delenv("COPERNICUS_CLIENT_ID", raising=False)
        monkeypatch.delenv("COPERNICUS_CLIENT_SECRET", raising=False)

        cache = TileCache(str(tmp_path / "cache"))
        with pytest.raises(RuntimeError, match="Copernicus credentials missing"):
            cache.get_tile(
                tile_id="T21NZG",
                date_range="2023-10-01/2023-10-31",
                bbox=SAINT_LAURENT_BBOX,
            )
