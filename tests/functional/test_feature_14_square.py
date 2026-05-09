"""Functional tests for Feature 14: Square Post-Processing.

These tests exercise the full square post-processing pipeline including
merging, grid overlay, confidence computation, thresholding, and GeoJSON export.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.data.square_postprocess import (
    create_square_grid,
    square_postprocess,
)

TARGET_CRS = "EPSG:2972"


def _make_probability_raster(
    tmp_path: Path,
    size: int = 256,
    values: np.ndarray | None = None,
    origin_x: float = 200_000.0,
    origin_y: float = 500_000.0,
    resolution: float = 10.0,
) -> Path:
    """Create a synthetic probability raster."""
    raster_path = tmp_path / f"probs_{size}.tif"
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


class TestFeature14SquareFlow:
    """End-to-end square post-processing workflow tests."""

    def test_full_postprocessing_pipeline(self, tmp_path: Path) -> None:
        """Mosaic -> grid -> threshold -> GeoJSON with squares."""
        probs = np.zeros((512, 512), dtype=np.float32)
        probs[100:400, 100:400] = 0.8
        raster_path = _make_probability_raster(tmp_path, size=512, values=probs)
        output_path = tmp_path / "detections.geojson"

        result = square_postprocess(
            [raster_path],
            grid_size_m=128.0,
            threshold=0.2,
            min_confidence=0.2,
            output_path=str(output_path),
        )

        assert result.exists()
        import geopandas as gpd

        gdf = gpd.read_file(result)
        assert len(gdf) > 0
        assert "detection_id" in gdf.columns
        assert "confidence" in gdf.columns
        assert "area_m2" in gdf.columns

        # All features should be perfect squares
        for geom in gdf.geometry:
            minx, miny, maxx, maxy = geom.bounds
            assert abs((maxx - minx) - (maxy - miny)) < 1e-6

    def test_squares_align_to_grid(self, tmp_path: Path) -> None:
        """Squares should align to the defined grid."""
        # Use an origin aligned to the grid so assertions are clean
        grid_size = 128.0
        origin_x = 200_000.0 - (200_000.0 % grid_size)
        origin_y = 500_000.0 - (500_000.0 % grid_size)
        probs = np.zeros((512, 512), dtype=np.float32)
        probs[100:400, 100:400] = 0.8
        raster_path = _make_probability_raster(
            tmp_path, size=512, values=probs, origin_x=origin_x, origin_y=origin_y
        )
        output_path = tmp_path / "aligned.geojson"

        square_postprocess(
            [raster_path],
            grid_size_m=grid_size,
            threshold=0.2,
            min_confidence=0.2,
            output_path=str(output_path),
        )

        import geopandas as gpd

        gdf = gpd.read_file(output_path)
        for geom in gdf.geometry:
            minx, miny, maxx, maxy = geom.bounds
            # Corners should align to grid_size multiples relative to the origin
            assert minx % grid_size < 1e-6 or abs(minx % grid_size - grid_size) < 1e-6
            assert miny % grid_size < 1e-6 or abs(miny % grid_size - grid_size) < 1e-6

    def test_confidence_correlates_with_probability(self, tmp_path: Path) -> None:
        """Higher probability -> higher confidence."""
        high_dir = tmp_path / "high"
        high_dir.mkdir(parents=True, exist_ok=True)
        probs_high = np.zeros((256, 256), dtype=np.float32)
        probs_high[50:200, 50:200] = 0.9
        raster_high = _make_probability_raster(high_dir, values=probs_high)

        low_dir = tmp_path / "low"
        low_dir.mkdir(parents=True, exist_ok=True)
        probs_low = np.zeros((256, 256), dtype=np.float32)
        probs_low[50:200, 50:200] = 0.3
        raster_low = _make_probability_raster(low_dir, values=probs_low)

        import geopandas as gpd

        out_high = tmp_path / "high.geojson"
        out_low = tmp_path / "low.geojson"

        square_postprocess(
            [raster_high],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(out_high),
        )
        square_postprocess(
            [raster_low],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(out_low),
        )

        gdf_high = gpd.read_file(out_high)
        gdf_low = gpd.read_file(out_low)

        assert len(gdf_high) > 0
        assert len(gdf_low) > 0
        assert gdf_high["confidence"].mean() > gdf_low["confidence"].mean()

    def test_different_grid_sizes_produce_different_counts(self, tmp_path: Path) -> None:
        """Smaller grid -> more cells -> potentially more detections."""
        probs = np.zeros((512, 512), dtype=np.float32)
        probs[100:400, 100:400] = 0.8
        raster_path = _make_probability_raster(tmp_path, size=512, values=probs)

        out_64 = tmp_path / "grid_64.geojson"
        out_128 = tmp_path / "grid_128.geojson"

        square_postprocess(
            [raster_path],
            grid_size_m=64.0,
            threshold=0.2,
            min_confidence=0.2,
            output_path=str(out_64),
        )
        square_postprocess(
            [raster_path],
            grid_size_m=128.0,
            threshold=0.2,
            min_confidence=0.2,
            output_path=str(out_128),
        )

        import geopandas as gpd

        gdf_64 = gpd.read_file(out_64)
        gdf_128 = gpd.read_file(out_128)

        # Smaller grid should produce at least as many cells, potentially more detections
        assert len(gdf_64) >= len(gdf_128)
