"""Tests for real data validation (Feature 1)."""

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


class TestValidateImage:
    """Test suite for image validation."""

    def test_valid_sentinel2_passes(self, synthetic_geotiff: Path) -> None:
        """Should return metadata dict for valid image."""
        result = validate_image(synthetic_geotiff)

        assert isinstance(result, dict)
        assert result["crs"] == TARGET_CRS
        assert result["band_count"] == 7
        assert result["width"] == 512
        assert result["height"] == 512
        assert result["resolution"] == 10.0
        assert hasattr(result["bounds"], "left")

    def test_wrong_band_count_raises(self, tmp_path: Path) -> None:
        """Should raise ValueError if band count != 9."""
        geotiff_path = tmp_path / "wrong_bands.tif"
        transform = from_origin(200_000.0, 510_000.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(6, 512, 512), dtype=np.uint16)

        with rasterio.open(
            geotiff_path,
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=6,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)

        with pytest.raises(ValueError, match="Expected 7 bands, got 6"):
            validate_image(geotiff_path)

    def test_missing_crs_raises(self, tmp_path: Path) -> None:
        """Should raise ValueError if CRS is undefined."""
        geotiff_path = tmp_path / "no_crs.tif"
        transform = from_origin(200_000.0, 510_000.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(7, 512, 512), dtype=np.uint16)

        with rasterio.open(
            geotiff_path,
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=7,
            dtype=data.dtype,
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(data)

        with pytest.raises(ValueError, match="CRS is undefined"):
            validate_image(geotiff_path)


class TestValidateLabels:
    """Test suite for label validation."""

    def test_reproject_to_target_crs(self, tmp_path: Path) -> None:
        """Should reproject WGS84 labels to EPSG:2972."""
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

        gdf = validate_labels(labels_path, expected_crs=TARGET_CRS)
        assert gdf.crs.to_string() == TARGET_CRS
        assert len(gdf) == 1

    def test_empty_labels_raises(self, empty_geopackage: Path) -> None:
        """Should raise ValueError for empty GeoPackage."""
        with pytest.raises(ValueError, match="No valid geometries found"):
            validate_labels(empty_geopackage, expected_crs=TARGET_CRS)


class TestSpatialOverlap:
    """Test suite for spatial overlap checking."""

    def test_full_overlap_returns_true(self) -> None:
        """Labels inside image bounds -> True."""
        image_bounds = type("Bounds", (), {"left": 0, "bottom": 0, "right": 1000, "top": 1000})
        polygon = Polygon([(100, 100), (900, 100), (900, 900), (100, 900)])
        gdf = gpd.GeoDataFrame({"label": ["mining"]}, geometry=[polygon], crs=TARGET_CRS)

        result = check_spatial_overlap(image_bounds, gdf)
        assert result["has_overlap"] is True
        assert result["total_labels"] == 1
        assert result["overlapping_labels"] == 1
        assert result["outside_labels"] == 0
        assert result["overlap_fraction"] > 0.5

    def test_no_overlap_returns_false(self) -> None:
        """Labels far from image -> False."""
        image_bounds = type("Bounds", (), {"left": 0, "bottom": 0, "right": 1000, "top": 1000})
        polygon = Polygon([(5000, 5000), (6000, 5000), (6000, 6000), (5000, 6000)])
        gdf = gpd.GeoDataFrame({"label": ["mining"]}, geometry=[polygon], crs=TARGET_CRS)

        result = check_spatial_overlap(image_bounds, gdf)
        assert result["has_overlap"] is False
        assert result["total_labels"] == 1
        assert result["overlapping_labels"] == 0
        assert result["outside_labels"] == 1
        assert result["overlap_fraction"] == 0.0
