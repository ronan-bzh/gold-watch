"""Functional tests for Feature 0: Copernicus Data Space Download.

These tests exercise the full auth-search-download workflow end-to-end,
using synthetic mocks so they run fast and offline.
"""

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

_AUTH_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


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


class TestFeature0DownloadFlow:
    """End-to-end download workflow tests."""

    def test_full_auth_search_download_flow(
        self, tmp_path: Path, requests_mock: Mocker
    ) -> None:
        """Auth -> search -> download produces a valid GeoTIFF."""
        # 1. Mock authentication
        requests_mock.post(
            _AUTH_URL,
            json={
                "access_token": "mock_token_12345",
                "token_type": "Bearer",
                "expires_in": 600,
            },
        )

        # 2. Mock STAC search
        requests_mock.post(
            _STAC_URL,
            json={
                "features": [
                    {
                        "id": "S2A_T22NLL_20230115T",
                        "bbox": [-54.1, 5.3, -53.9, 5.5],
                        "properties": {
                            "datetime": "2023-01-15T10:00:00Z",
                            "eo:cloud_cover": 5.2,
                        },
                    }
                ]
            },
        )

        # 3. Mock Process API download
        requests_mock.post(
            _PROCESS_URL,
            content=_make_geotiff_bytes(count=7),
        )

        # Execute full workflow
        token = get_access_token("valid_id", "valid_secret")
        scenes = search_scenes(
            bbox=(-54.1, 5.3, -53.9, 5.5),
            date_range="2023-01-01/2023-01-31",
            token=token,
        )
        assert len(scenes) == 1

        output_path = tmp_path / "scene.tif"
        result = download_scene(
            scenes[0],
            token,
            output_path,
            bands=["B02", "B03", "B04", "B08", "B11", "B12"],
            resolution=10,
            bbox=(-54.1, 5.3, -53.9, 5.5),
        )

        assert result.exists()
        with rasterio.open(result) as src:
            assert src.count == 7  # 6 requested + SCL appended
            assert src.crs is not None
            assert src.width == 64
            assert src.height == 64

    def test_search_cloud_filter_affects_results(
        self, requests_mock: Mocker
    ) -> None:
        """STAC search respects max_cloud_cover threshold."""
        requests_mock.post(
            _STAC_URL,
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
                ]
            },
        )

        items = search_scenes(
            bbox=(-54.1, 5.3, -53.9, 5.5),
            date_range="2023-01-01/2023-01-31",
            token="mock_token",
            max_cloud_cover=20.0,
        )
        assert len(items) == 1
        assert items[0]["id"] == "S2A_low"

    def test_download_custom_bands_in_evalscript(
        self, tmp_path: Path, requests_mock: Mocker
    ) -> None:
        """Custom band selection propagates into the Process API payload."""
        requests_mock.post(
            _PROCESS_URL,
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

        output_path = tmp_path / "scene.tif"
        download_scene(
            item,
            "mock_token",
            output_path,
            bands=["B02", "B03", "B04"],
            bbox=(-54.1, 5.3, -53.9, 5.5),
        )

        history = requests_mock.request_history
        process_requests = [
            r for r in history if r.url == _PROCESS_URL
        ]
        assert len(process_requests) == 1
        payload = process_requests[0].json()
        assert '"B02"' in payload["evalscript"]
        assert '"B03"' in payload["evalscript"]
        assert '"B04"' in payload["evalscript"]
        assert '"SCL"' in payload["evalscript"]
        assert '"B08"' not in payload["evalscript"]

    def test_download_custom_resolution_affects_size(
        self, tmp_path: Path, requests_mock: Mocker
    ) -> None:
        """Resolution parameter changes requested width/height in payload."""
        requests_mock.post(
            _PROCESS_URL,
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

        output_10m = tmp_path / "scene_10m.tif"
        download_scene(
            item,
            "mock_token",
            output_10m,
            resolution=10,
            bbox=(-54.1, 5.3, -53.9, 5.5),
        )

        output_20m = tmp_path / "scene_20m.tif"
        download_scene(
            item,
            "mock_token",
            output_20m,
            resolution=20,
            bbox=(-54.1, 5.3, -53.9, 5.5),
        )

        history = requests_mock.request_history
        process_requests = [
            r for r in history if r.url == _PROCESS_URL
        ]
        assert len(process_requests) == 2

        payload_10m = process_requests[0].json()
        payload_20m = process_requests[1].json()

        # At 20m resolution the same bbox should yield roughly half the pixels of 10m
        assert payload_20m["output"]["width"] < payload_10m["output"]["width"]
        assert payload_20m["output"]["height"] < payload_10m["output"]["height"]

    def test_auth_failure_raises_authentication_error(
        self, requests_mock: Mocker
    ) -> None:
        """Invalid credentials raise AuthenticationError."""
        requests_mock.post(_AUTH_URL, status_code=401, json={"error": "invalid_client"})

        with pytest.raises(AuthenticationError):
            get_access_token("bad_id", "bad_secret")

    def test_empty_search_raises_runtime_error(
        self, requests_mock: Mocker
    ) -> None:
        """No matching scenes raise RuntimeError."""
        requests_mock.post(
            _STAC_URL,
            json={"features": []},
        )

        with pytest.raises(RuntimeError, match="No scenes found"):
            search_scenes(
                bbox=(-54.1, 5.3, -53.9, 5.5),
                date_range="2023-01-01/2023-01-31",
                token="mock_token",
            )
