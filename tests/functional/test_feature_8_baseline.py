"""Functional tests for Feature 8: Spectral Rule-Based Baseline.

These tests exercise the rule-based detection workflow end-to-end,
including NDVI/BSI computation and polygon extraction.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from goldmine_watch.baseline.rules import (
    compute_ndvi,
    detect_mining_rules,
    rules_to_polygons,
)

TARGET_CRS = "EPSG:2972"


def _make_multiband_image(tmp_path: Path, size: int = 256) -> Path:
    """Create a synthetic 6-band image (B02, B03, B04, B08, B11, B12)."""
    image_path = tmp_path / "multiband.tif"
    transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)

    # Simulate: vegetation = high NIR, low RED; bare soil = low NIR, high RED+SWIR
    data = np.zeros((6, size, size), dtype=np.uint16)
    data[0] = 500   # B02
    data[1] = 600   # B03
    data[2] = 800   # B04  (RED)
    data[3] = 2000  # B08  (NIR) — vegetation-like by default
    data[4] = 1200  # B11  (SWIR1)
    data[5] = 1000  # B12  (SWIR2)

    # Bare soil patch
    data[2, 50:150, 50:150] = 3000   # high RED
    data[3, 50:150, 50:150] = 400    # low NIR
    data[4, 50:150, 50:150] = 2500   # high SWIR1

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

    return image_path


class TestFeature8BaselineFlow:
    """End-to-end rule-based baseline workflow tests."""

    def test_detect_mining_rules_produces_mask(
        self, tmp_path: Path
    ) -> None:
        """Running rules on a multiband image produces a binary mask."""
        image_path = _make_multiband_image(tmp_path)
        mask = detect_mining_rules(image_path, ndvi_threshold=0.2, bsi_threshold=0.1)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        # Bare soil patch should be detected
        assert mask[50:150, 50:150].sum() > 0

    def test_ndvi_range_between_minus_one_and_one(
        self, tmp_path: Path
    ) -> None:
        """NDVI values should be in [-1, 1]."""
        image_path = _make_multiband_image(tmp_path)
        with rasterio.open(image_path) as src:
            red = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)

        ndvi = compute_ndvi(nir, red)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_bare_soil_detected_low_ndvi_high_bsi(
        self, tmp_path: Path
    ) -> None:
        """Bare soil pixels (low NIR, high RED+SWIR) should be flagged."""
        image_path = _make_multiband_image(tmp_path)
        mask = detect_mining_rules(image_path, ndvi_threshold=0.2, bsi_threshold=0.1)

        # The synthetic bare-soil patch is at [50:150, 50:150]
        assert mask[50:150, 50:150].sum() > 0

    def test_forest_not_detected_high_ndvi(
        self, tmp_path: Path
    ) -> None:
        """Vegetation pixels (high NIR, low RED) should NOT be flagged."""
        image_path = _make_multiband_image(tmp_path)
        mask = detect_mining_rules(image_path, ndvi_threshold=0.2, bsi_threshold=0.1)

        # Background is vegetation-like
        assert mask[0:50, 0:50].sum() == 0

    def test_rules_to_polygons_returns_geodataframe(
        self, tmp_path: Path
    ) -> None:
        """Converting a rule mask to polygons yields a GeoDataFrame."""
        image_path = _make_multiband_image(tmp_path)
        mask = detect_mining_rules(image_path, ndvi_threshold=0.2, bsi_threshold=0.1)

        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs

        gdf = rules_to_polygons(mask, transform, crs)
        assert gdf is not None
        assert len(gdf) > 0
        assert gdf.crs is not None

    def test_empty_mask_produces_empty_geodataframe(
        self, tmp_path: Path
    ) -> None:
        """All-zero mask should produce an empty GeoDataFrame."""
        mask = np.zeros((128, 128), dtype=np.uint8)
        transform = from_origin(0, 1280, 10, 10)
        gdf = rules_to_polygons(mask, transform, TARGET_CRS)
        assert len(gdf) == 0

    def test_threshold_tuning_changes_mask(
        self, tmp_path: Path
    ) -> None:
        """Stricter threshold should produce fewer detections."""
        image_path = _make_multiband_image(tmp_path)
        mask_loose = detect_mining_rules(image_path, ndvi_threshold=0.5, bsi_threshold=-0.5)
        mask_strict = detect_mining_rules(image_path, ndvi_threshold=0.1, bsi_threshold=0.3)

        assert mask_strict.sum() <= mask_loose.sum()
