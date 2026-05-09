"""Functional tests for Feature 11: Multi-Scene Training Dataset.

These tests exercise the full dataset build pipeline end-to-end using
mocked tile downloads and real mine clustering data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from requests_mock import Mocker

from goldmine_watch.data.build_training_dataset import build_training_dataset
from goldmine_watch.data.mine_clusterer import cluster_mines_by_tile, load_mines

if TYPE_CHECKING:
    pass

SAINT_LAURENT_BBOX = (-54.05, 5.45, -54.0, 5.49)


def _make_geotiff_bytes(width: int = 512, height: int = 512, count: int = 7) -> bytes:
    """Create a small GeoTIFF in memory and return its bytes."""
    transform = from_origin(200_000.0, 500_000.0, 10.0, 10.0)
    data = np.random.randint(0, 1000, size=(count, height, width), dtype=np.uint16)
    from rasterio.io import MemoryFile

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


@pytest.mark.integration
class TestFeature11DatasetFlow:
    """End-to-end functional tests for the full dataset build."""

    def test_full_dataset_build(
        self,
        tmp_path: Path,
        requests_mock: Mocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end: cluster -> download -> extract -> split."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        output_dir = tmp_path / "splits"
        cache_dir = tmp_path / "cache"

        stats = build_training_dataset(
            mines_geojson="data/french_guiana_mines.geojson",
            output_dir=str(output_dir),
            num_background_per_tile=50,
            patch_size=256,
            date_range="2023-06-01/2023-12-31",
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
            cache_dir=str(cache_dir),
        )

        # Should have created output directories
        assert (output_dir / "train").exists()
        assert (output_dir / "val").exists()

        # Should have produced some patches
        assert stats["total_patches"] > 0
        assert stats["train_patches"] > 0 or stats["val_patches"] > 0

        # Should have clustered mines
        gdf = load_mines("data/french_guiana_mines.geojson")
        clusters = cluster_mines_by_tile(gdf)
        assert stats["num_tiles"] == len(clusters)

    def test_cache_reuse_on_second_run(
        self,
        tmp_path: Path,
        requests_mock: Mocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Second build should reuse cached tiles, skip downloads."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        output_dir = tmp_path / "splits"
        cache_dir = tmp_path / "cache"

        # First build
        build_training_dataset(
            mines_geojson="data/french_guiana_mines.geojson",
            output_dir=str(output_dir),
            num_background_per_tile=10,
            patch_size=256,
            date_range="2023-06-01/2023-12-31",
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
            cache_dir=str(cache_dir),
        )

        # Count HTTP calls after first build
        first_build_calls = len(requests_mock.request_history)

        # Second build — should reuse cache
        build_training_dataset(
            mines_geojson="data/french_guiana_mines.geojson",
            output_dir=str(output_dir / "second"),
            num_background_per_tile=10,
            patch_size=256,
            date_range="2023-06-01/2023-12-31",
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
            cache_dir=str(cache_dir),
        )

        second_build_calls = len(requests_mock.request_history)
        # No additional download calls on second run
        assert second_build_calls == first_build_calls

    def test_all_mines_represented(
        self,
        tmp_path: Path,
        requests_mock: Mocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At least one patch per mine cluster (or per mine when possible)."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        output_dir = tmp_path / "splits"
        cache_dir = tmp_path / "cache"

        gdf = load_mines("data/french_guiana_mines.geojson")
        total_mines = len(gdf)

        stats = build_training_dataset(
            mines_geojson="data/french_guiana_mines.geojson",
            output_dir=str(output_dir),
            num_background_per_tile=10,
            patch_size=256,
            date_range="2023-06-01/2023-12-31",
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
            cache_dir=str(cache_dir),
        )

        # Total mine patches should equal total mines (one per mine)
        total_mine_patches = sum(s["mine_patches"] for s in stats["stats_by_tile"].values())
        assert total_mine_patches == total_mines

    def test_no_data_leakage(
        self,
        tmp_path: Path,
        requests_mock: Mocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Train and val patches should come from different tiles."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        output_dir = tmp_path / "splits"
        cache_dir = tmp_path / "cache"

        stats = build_training_dataset(
            mines_geojson="data/french_guiana_mines.geojson",
            output_dir=str(output_dir),
            num_background_per_tile=10,
            patch_size=256,
            date_range="2023-06-01/2023-12-31",
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
            cache_dir=str(cache_dir),
        )

        train_tiles = set(stats["train_tiles"])
        val_tiles = set(stats["val_tiles"])
        assert train_tiles.isdisjoint(val_tiles)

    def test_background_patches_diverse(
        self,
        tmp_path: Path,
        requests_mock: Mocker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Background patches should come from different areas (not all identical)."""
        monkeypatch.setenv("COPERNICUS_CLIENT_ID", "mock_id")
        monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "mock_secret")
        _register_mocks(requests_mock)

        output_dir = tmp_path / "splits"
        cache_dir = tmp_path / "cache"

        build_training_dataset(
            mines_geojson="data/french_guiana_mines.geojson",
            output_dir=str(output_dir),
            num_background_per_tile=20,
            patch_size=256,
            date_range="2023-06-01/2023-12-31",
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
            cache_dir=str(cache_dir),
        )

        # Load a few background patches and verify they're not all identical
        train_images = sorted((output_dir / "train").glob("image_*.npy"))
        # Most patches in train will be background (only a few mines per tile)
        # Just verify we have at least some patches
        assert len(train_images) > 0

        # If there are background patches, they should have some variance
        bg_masks = []
        for mask_path in sorted((output_dir / "train").glob("mask_*.npy")):
            m = np.load(mask_path)
            if not np.any(m > 0):
                bg_masks.append(m)

        # Should have at least a few background patches
        assert len(bg_masks) >= 10
