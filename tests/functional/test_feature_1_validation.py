"""Functional tests for Feature 1: Real Data Ingestion & Validation.

These tests exercise the full image + labels validation pipeline,
including reprojection and spatial overlap checks.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.data.validate import (
    check_spatial_overlap,
    validate_image,
    validate_labels,
)

TARGET_CRS = "EPSG:2972"
WGS84_CRS = "EPSG:4326"


class TestFeature1ValidationFlow:
    """End-to-end validation workflow tests."""

    def test_validate_image_then_labels(
        self, synthetic_geotiff: Path, synthetic_labels: Path
    ) -> None:
        """Sequential validation returns correct metadata for both assets."""
        img_meta = validate_image(synthetic_geotiff)
        assert isinstance(img_meta, dict)
        assert img_meta["crs"] == TARGET_CRS
        assert img_meta["band_count"] == 7
        assert img_meta["width"] == 512
        assert img_meta["height"] == 512

        labels_gdf = validate_labels(synthetic_labels, expected_crs=TARGET_CRS)
        assert isinstance(labels_gdf, gpd.GeoDataFrame)
        assert labels_gdf.crs.to_string() == TARGET_CRS
        assert len(labels_gdf) == 3

    def test_labels_reprojected_to_image_crs(self, tmp_path: Path) -> None:
        """WGS84 labels are silently reprojected to the image CRS."""
        # Create a synthetic image in EPSG:2972
        image_path = tmp_path / "image.tif"
        transform = from_origin(200_000.0, 510_000.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(7, 512, 512), dtype=np.uint16)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=7,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)

        # Create WGS84 labels
        labels_path = tmp_path / "labels_wgs84.gpkg"
        polygon = Polygon([
            (-52.3, 4.9),
            (-52.2, 4.9),
            (-52.2, 5.0),
            (-52.3, 5.0),
        ])
        gdf_wgs84 = gpd.GeoDataFrame(
            {"label": ["mining"]},
            geometry=[polygon],
            crs=WGS84_CRS,
        )
        gdf_wgs84.to_file(labels_path, driver="GPKG")

        labels_gdf = validate_labels(labels_path, expected_crs=TARGET_CRS)
        assert labels_gdf.crs.to_string() == TARGET_CRS
        assert len(labels_gdf) == 1

    def test_full_spatial_overlap(self, tmp_path: Path) -> None:
        """All labels inside image bounds -> has_overlap=True, fraction=1.0."""
        image_path = tmp_path / "image.tif"
        transform = from_origin(0, 1000, 10, 10)
        data = np.random.randint(0, 10_000, size=(7, 100, 100), dtype=np.uint16)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=7,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)

        labels_path = tmp_path / "labels.gpkg"
        polygon = Polygon([(10, 10), (900, 10), (900, 900), (10, 900)])
        gdf = gpd.GeoDataFrame(
            {"label": ["mining"]},
            geometry=[polygon],
            crs=TARGET_CRS,
        )
        gdf.to_file(labels_path, driver="GPKG")

        img_meta = validate_image(image_path)
        labels_gdf = validate_labels(labels_path, expected_crs=TARGET_CRS)
        overlap = check_spatial_overlap(img_meta["bounds"], labels_gdf)

        assert overlap["has_overlap"] is True
        assert overlap["total_labels"] == 1
        assert overlap["overlapping_labels"] == 1
        assert overlap["outside_labels"] == 0
        assert overlap["overlap_fraction"] == 1.0

    def test_no_spatial_overlap(self, tmp_path: Path) -> None:
        """Labels far outside image bounds -> has_overlap=False."""
        image_path = tmp_path / "image.tif"
        transform = from_origin(0, 1000, 10, 10)
        data = np.random.randint(0, 10_000, size=(7, 100, 100), dtype=np.uint16)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=7,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)

        labels_path = tmp_path / "labels.gpkg"
        polygon = Polygon([(5000, 5000), (6000, 5000), (6000, 6000), (5000, 6000)])
        gdf = gpd.GeoDataFrame(
            {"label": ["mining"]},
            geometry=[polygon],
            crs=TARGET_CRS,
        )
        gdf.to_file(labels_path, driver="GPKG")

        img_meta = validate_image(image_path)
        labels_gdf = validate_labels(labels_path, expected_crs=TARGET_CRS)
        overlap = check_spatial_overlap(img_meta["bounds"], labels_gdf)

        assert overlap["has_overlap"] is False
        assert overlap["overlapping_labels"] == 0
        assert overlap["outside_labels"] == 1
        assert overlap["overlap_fraction"] == 0.0

    def test_missing_inputs_raise(self, tmp_path: Path) -> None:
        """Nonexistent image or label paths raise clear errors."""
        missing_image = tmp_path / "does_not_exist.tif"
        missing_labels = tmp_path / "does_not_exist.gpkg"

        with pytest.raises((ValueError, rasterio.errors.RasterioIOError)):
            validate_image(missing_image)

        # GeoPandas I/O backend may raise pyogrio.errors.DataSourceError or FileNotFoundError
        try:
            from pyogrio.errors import DataSourceError

            expected_exceptions: tuple[type[BaseException], ...] = (ValueError, DataSourceError)
        except ImportError:
            expected_exceptions = (ValueError, FileNotFoundError)

        with pytest.raises(expected_exceptions):
            validate_labels(missing_labels, expected_crs=TARGET_CRS)
