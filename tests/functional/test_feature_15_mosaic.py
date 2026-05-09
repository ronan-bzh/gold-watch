"""Functional tests for Feature 15: Mosaic Builder.

These tests exercise the full mosaic workflow including multi-raster merging,
CRS preservation, and value-range validation.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from goldmine_watch.data.mosaic import build_mosaic, validate_mosaic

TARGET_CRS = "EPSG:2972"


def _make_probability_raster(
    tmp_path: Path,
    size: int = 256,
    values: np.ndarray | None = None,
    origin_x: float = 200_000.0,
    origin_y: float = 500_000.0,
    resolution: float = 10.0,
) -> Path:
    """Create a synthetic probability GeoTIFF."""
    raster_path = tmp_path / f"prob_{origin_x}_{origin_y}.tif"
    probs = np.zeros((size, size), dtype=np.float32) if values is None else values
    transform = from_origin(origin_x, origin_y + size * resolution, resolution, resolution)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype=probs.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        dst.write(probs, 1)

    return raster_path


class TestFeature15MosaicFlow:
    """End-to-end mosaic workflow tests."""

    def test_full_mosaic_build(self, tmp_path: Path) -> None:
        """Merge all tile predictions into one mosaic."""
        rasters: list[Path] = []
        for i in range(3):
            probs = np.random.rand(128, 128).astype(np.float32)
            origin_x = 200_000.0 + i * 1_280.0
            r = _make_probability_raster(tmp_path, values=probs, origin_x=origin_x)
            rasters.append(r)

        output_path = tmp_path / "mosaic.tif"
        result = build_mosaic(rasters, output_path=str(output_path), method="mean")

        assert result.exists()
        with rasterio.open(result) as src:
            assert src.count == 1
            assert src.width > 0
            assert src.height > 0

    def test_mosaic_has_correct_crs(self, tmp_path: Path) -> None:
        """Mosaic CRS should match input tiles."""
        probs = np.random.rand(128, 128).astype(np.float32)
        raster_path = _make_probability_raster(tmp_path, values=probs)

        output_path = tmp_path / "mosaic.tif"
        result = build_mosaic([raster_path], output_path=str(output_path))

        with rasterio.open(result) as src:
            assert str(src.crs) == TARGET_CRS

    def test_mosaic_values_in_valid_range(self, tmp_path: Path) -> None:
        """All pixel values should be [0, 1]."""
        probs1 = np.random.rand(128, 128).astype(np.float32)
        probs2 = np.random.rand(128, 128).astype(np.float32)
        r1 = _make_probability_raster(tmp_path, values=probs1, origin_x=200_000.0)
        r2 = _make_probability_raster(tmp_path, values=probs2, origin_x=200_640.0)

        output_path = tmp_path / "mosaic.tif"
        result = build_mosaic([r1, r2], output_path=str(output_path), method="mean")

        report = validate_mosaic(result)
        assert report["min_value"] >= 0.0
        assert report["max_value"] <= 1.0
        assert report["out_of_range_count"] == 0
