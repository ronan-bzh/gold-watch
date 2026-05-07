"""Tests for data ingestion and config validation."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from shapely.geometry import Polygon

from goldmine_watch.config import load_config
from goldmine_watch.data.ingest import burn_mask, load_labels

TARGET_CRS = "EPSG:2972"


class TestConfigValidation:
    """Test suite for configuration loading and validation."""

    def test_valid_config_passes(self, tmp_path: Path) -> None:
        """Should load a valid config and expose typed fields."""
        config = {
            "geospatial": {"patch_size": 256},
            "data": {"max_cloud_cover": 20},
        }
        config_path = tmp_path / "valid.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f)

        cfg = load_config(config_path)
        assert cfg.geospatial.patch_size == 256
        assert cfg.data.max_cloud_cover == 20

    def test_invalid_config_raises(self, tmp_path: Path) -> None:
        """Should raise ValueError with a clear message for invalid fields."""
        config = {
            "geospatial": {"patch_size": -1},
            "data": {"max_cloud_cover": 150},
        }
        config_path = tmp_path / "invalid.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError):
            load_config(config_path)


class TestLoadLabels:
    """Test suite for label loading and validation."""

    def test_load_valid_geopackage(self, synthetic_labels: Path) -> None:
        """Should load a valid GeoPackage file and return a GeoDataFrame."""
        gdf = load_labels(synthetic_labels)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert gdf.crs.to_string() == TARGET_CRS

    def test_skip_invalid_geometries(self, synthetic_labels_with_invalid: Path) -> None:
        """Should skip invalid geometries with a warning instead of crashing."""
        gdf = load_labels(synthetic_labels_with_invalid)
        # Only the valid polygon should remain
        assert len(gdf) == 1
        assert all(gdf.geometry.is_valid)

    def test_reproject_to_target_crs(self, tmp_path: Path) -> None:
        """Should reproject labels to the pipeline-wide target CRS."""
        # Create a label in WGS84
        polygon = Polygon(
            [
                (-52.3, 4.9),
                (-52.2, 4.9),
                (-52.2, 5.0),
                (-52.3, 5.0),
            ]
        )
        gdf_wgs84 = gpd.GeoDataFrame(
            {"label": ["mining"]},
            geometry=[polygon],
            crs="EPSG:4326",
        )
        labels_path = tmp_path / "labels_wgs84.gpkg"
        gdf_wgs84.to_file(labels_path, driver="GPKG")

        gdf = load_labels(labels_path, target_crs=TARGET_CRS)
        assert gdf.crs.to_string() == TARGET_CRS
        assert len(gdf) == 1

    def test_empty_file_raises(self, empty_geopackage: Path) -> None:
        """Should raise ValueError when the label file is empty."""
        with pytest.raises(ValueError, match="No valid geometries found"):
            load_labels(empty_geopackage)


class TestBurnMask:
    """Test suite for rasterizing labels into binary masks."""

    def test_burn_mask_shape(self, synthetic_labels: Path, synthetic_geotiff: Path) -> None:
        """Should produce a binary mask with the same shape as the reference raster."""
        gdf = load_labels(synthetic_labels)
        mask = burn_mask(gdf, synthetic_geotiff)

        with rasterio.open(synthetic_geotiff) as src:
            expected_shape = (src.height, src.width)

        assert mask.shape == expected_shape
        assert mask.dtype == np.uint8
        assert mask.min() == 0
        assert mask.max() == 1
        # At least some pixels should be labelled
        assert mask.sum() > 0
