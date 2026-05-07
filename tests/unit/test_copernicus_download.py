"""Tests for Copernicus Data Space download functionality."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from requests_mock import Mocker

from goldmine_watch.data.copernicus import (
    AuthenticationError,
    download_scene,
    get_access_token,
    search_scenes,
)


def _make_geotiff_bytes(
    width: int = 64, height: int = 64, count: int = 1, dtype: str = "uint16"
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


class TestGetAccessToken:
    """Test suite for Copernicus OAuth authentication."""

    def test_token_returns_string(self, requests_mock: Mocker) -> None:
        """Should return a non-empty bearer token on valid credentials."""
        requests_mock.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            json={
                "access_token": "mock_token_12345",
                "token_type": "Bearer",
                "expires_in": 600,
            },
        )

        token = get_access_token("valid_id", "valid_secret")
        assert isinstance(token, str)
        assert token == "mock_token_12345"

    def test_invalid_credentials_raises(self, requests_mock: Mocker) -> None:
        """Should raise AuthenticationError on 401."""
        requests_mock.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            status_code=401,
            json={"error": "invalid_client"},
        )

        with pytest.raises(AuthenticationError):
            get_access_token("bad_id", "bad_secret")


class TestSearchScenes:
    """Test suite for STAC catalog search."""

    def test_search_returns_items(self, requests_mock: Mocker) -> None:
        """Should return list of STAC items for a valid query."""
        requests_mock.post(
            "https://catalogue.dataspace.copernicus.eu/stac/search",
            json={
                "features": [
                    {
                        "id": "S2A_T22NLL_20230115T",
                        "properties": {"eo:cloud_cover": 5.2},
                    },
                    {
                        "id": "S2B_T22NLL_20230110T",
                        "properties": {"eo:cloud_cover": 12.8},
                    },
                ]
            },
        )

        items = search_scenes(
            bbox=(-54.1, 5.3, -53.9, 5.5),
            date_range="2023-01-01/2023-01-31",
            token="mock_token",
        )
        assert len(items) == 2
        assert items[0]["id"] == "S2A_T22NLL_20230115T"

    def test_no_scenes_raises(self, requests_mock: Mocker) -> None:
        """Should raise RuntimeError when catalog returns empty."""
        requests_mock.post(
            "https://catalogue.dataspace.copernicus.eu/stac/search",
            json={"features": []},
        )

        with pytest.raises(RuntimeError, match="No scenes found"):
            search_scenes(
                bbox=(-54.1, 5.3, -53.9, 5.5),
                date_range="2023-01-01/2023-01-31",
                token="mock_token",
            )

    def test_cloud_cover_filter(self, requests_mock: Mocker) -> None:
        """Should exclude scenes with cloud cover above threshold."""
        requests_mock.post(
            "https://catalogue.dataspace.copernicus.eu/stac/search",
            json={
                "features": [
                    {
                        "id": "S2A_low",
                        "properties": {"eo:cloud_cover": 5.0},
                    },
                    {
                        "id": "S2A_high",
                        "properties": {"eo:cloud_cover": 50.0},
                    },
                    {
                        "id": "S2B_mid",
                        "properties": {"eo:cloud_cover": 15.0},
                    },
                ]
            },
        )

        items = search_scenes(
            bbox=(-54.1, 5.3, -53.9, 5.5),
            date_range="2023-01-01/2023-01-31",
            token="mock_token",
            max_cloud_cover=20.0,
        )
        assert len(items) == 2
        assert items[0]["id"] == "S2A_low"
        assert items[1]["id"] == "S2B_mid"


class TestDownloadScene:
    """Test suite for scene download via Sentinel Hub Process API."""

    def test_download_writes_geotiff(self, tmp_path: Path, requests_mock: Mocker) -> None:
        """Should create a valid GeoTIFF at output_path."""
        requests_mock.post(
            "https://sh.dataspace.copernicus.eu/api/v1/process",
            content=_make_geotiff_bytes(count=7),
        )

        item = {
            "id": "S2A_T22NLL_20230115T",
            "bbox": [-54.1, 5.3, -53.9, 5.5],
            "properties": {
                "datetime": "2023-01-15T10:00:00Z",
                "eo:cloud_cover": 5.2,
            },
        }

        output_path = tmp_path / "scene.tif"
        result = download_scene(item, "mock_token", output_path)

        assert result.exists()
        with rasterio.open(result) as src:
            assert src.count == 7
            assert src.crs is not None
            assert src.width == 64
            assert src.height == 64

    def test_band_selection(self, tmp_path: Path, requests_mock: Mocker) -> None:
        """Should only download requested bands."""
        requests_mock.post(
            "https://sh.dataspace.copernicus.eu/api/v1/process",
            content=_make_geotiff_bytes(count=4),
        )

        item = {
            "id": "S2A_T22NLL_20230115T",
            "bbox": [-54.1, 5.3, -53.9, 5.5],
            "properties": {
                "datetime": "2023-01-15T10:00:00Z",
                "eo:cloud_cover": 5.2,
            },
        }

        output_path = tmp_path / "scene_rgb.tif"
        result = download_scene(item, "mock_token", output_path, bands=["B02", "B03", "B04"])

        assert result.exists()
        with rasterio.open(result) as src:
            # 3 requested + SCL auto-appended = 4 bands
            assert src.count == 4

        # Verify the Process API payload contained the right bands
        history = requests_mock.request_history
        process_requests = [
            r for r in history if r.url == "https://sh.dataspace.copernicus.eu/api/v1/process"
        ]
        assert len(process_requests) == 1
        payload = process_requests[0].json()
        assert '"B02"' in payload["evalscript"]
        assert '"B03"' in payload["evalscript"]
        assert '"B04"' in payload["evalscript"]
        assert '"SCL"' in payload["evalscript"]
        assert '"B08"' not in payload["evalscript"]
        assert "units" in payload["evalscript"]
