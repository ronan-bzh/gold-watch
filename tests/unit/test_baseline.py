"""Unit tests for spectral rule-based baseline."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from goldmine_watch.baseline.rules import (
    compute_ndvi,
    detect_mining_rules,
    rules_to_polygons,
)

TARGET_CRS = "EPSG:2972"


class TestBaselineRules:
    """Tests for spectral rule-based detection."""

    def test_ndvi_range(self) -> None:
        """NDVI should be between -1 and 1."""
        nir = np.array([100, 200, 300], dtype=np.float32)
        red = np.array([300, 200, 100], dtype=np.float32)
        ndvi = compute_ndvi(nir, red)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0
        assert ndvi[0] == pytest.approx(-0.5, abs=1e-6)
        assert ndvi[2] == pytest.approx(0.5, abs=1e-6)

    def test_bare_soil_detected(self, tmp_path: Path) -> None:
        """Pixel with low NDVI and high BSI should be flagged."""
        image_path = tmp_path / "bare_soil.tif"
        size = 64
        transform = from_origin(
            200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0
        )
        data = np.zeros((6, size, size), dtype=np.uint16)
        data[0] = 500  # B02
        data[1] = 600  # B03
        data[2] = 800  # B04  (RED)
        data[3] = 2000  # B08  (NIR)
        data[4] = 1200  # B11  (SWIR1)
        data[5] = 1000  # B12  (SWIR2)
        # Inject a bare-soil patch
        data[2, 20:40, 20:40] = 3000  # high RED
        data[3, 20:40, 20:40] = 400  # low NIR
        data[4, 20:40, 20:40] = 2500  # high SWIR1

        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=6,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)
            dst.set_band_description(1, "B02")
            dst.set_band_description(2, "B03")
            dst.set_band_description(3, "B04")
            dst.set_band_description(4, "B08")
            dst.set_band_description(5, "B11")
            dst.set_band_description(6, "B12")

        mask = detect_mining_rules(
            image_path, ndvi_threshold=0.2, bsi_threshold=0.1
        )
        assert mask[20:40, 20:40].sum() > 0

    def test_forest_not_detected(self, tmp_path: Path) -> None:
        """Pixel with high NDVI should NOT be flagged."""
        image_path = tmp_path / "forest.tif"
        size = 64
        transform = from_origin(
            200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0
        )
        data = np.zeros((6, size, size), dtype=np.uint16)
        data[0] = 500  # B02
        data[1] = 600  # B03
        data[2] = 800  # B04  (RED)
        data[3] = 2000  # B08  (NIR)
        data[4] = 1200  # B11  (SWIR1)
        data[5] = 1000  # B12  (SWIR2)

        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=6,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)
            dst.set_band_description(1, "B02")
            dst.set_band_description(2, "B03")
            dst.set_band_description(3, "B04")
            dst.set_band_description(4, "B08")
            dst.set_band_description(5, "B11")
            dst.set_band_description(6, "B12")

        mask = detect_mining_rules(
            image_path, ndvi_threshold=0.2, bsi_threshold=0.1
        )
        # Background is vegetation-like (high NIR, low RED)
        assert mask[0:20, 0:20].sum() == 0

    def test_rules_produce_polygons(self) -> None:
        """rules_to_polygons should return a non-empty GeoDataFrame."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        transform = from_origin(
            200_000.0, 500_000.0 + 64 * 10.0, 10.0, 10.0
        )
        gdf = rules_to_polygons(mask, transform, TARGET_CRS)
        assert gdf is not None
        assert len(gdf) > 0
        assert gdf.crs is not None
