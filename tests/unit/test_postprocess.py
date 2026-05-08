"""Unit tests for post-processing."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from goldmine_watch.inference.postprocess import postprocess

TARGET_CRS = "EPSG:2972"


def _make_probability_raster(
    tmp_path: Path, size: int = 256, values: np.ndarray | None = None
) -> Path:
    """Create a synthetic probability raster."""
    raster_path = tmp_path / "probs.tif"
    probs = np.zeros((size, size), dtype=np.float32) if values is None else values
    transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)

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


class TestPostprocess:
    """Tests for raster-to-vector post-processing."""

    def test_threshold_creates_binary_mask(self, tmp_path: Path) -> None:
        """Values >= 0.5 become polygons; values < 0.5 do not."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8
        probs[180:200, 180:200] = 0.3
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "polygons.gpkg"

        result = postprocess(raster_path, output_path, threshold=0.5, min_area_pixels=1)

        gdf = gpd.read_file(result)
        assert len(gdf) == 1
        # The 0.8 square should be detected
        assert gdf.iloc[0]["area_m2"] > 0

    def test_small_polygons_filtered(self, tmp_path: Path) -> None:
        """Polygon with area < min_area_pixels should be removed."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8  # 100x100 = 10_000 px
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "polygons_filtered.gpkg"

        result = postprocess(raster_path, output_path, threshold=0.5, min_area_pixels=11_000)

        gdf = gpd.read_file(result)
        assert len(gdf) == 0

    def test_output_has_expected_columns(self, tmp_path: Path) -> None:
        """GeoPackage should have geometry + detection_id, confidence, area_m2, area_ha."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "polygons_cols.gpkg"

        result = postprocess(raster_path, output_path, threshold=0.5, min_area_pixels=1)

        gdf = gpd.read_file(result)
        assert "detection_id" in gdf.columns
        assert "confidence" in gdf.columns
        assert "area_m2" in gdf.columns
        assert "area_ha" in gdf.columns
        assert gdf.iloc[0]["detection_id"] == 1
        assert gdf.iloc[0]["confidence"] > 0
        assert gdf.iloc[0]["area_m2"] > 0
        assert gdf.iloc[0]["area_ha"] > 0

    def test_min_area_m2_filter(self, tmp_path: Path) -> None:
        """min_area_m2 should remove polygons below the threshold."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:60, 50:60] = 0.8  # 10x10 px @ 10m = 100m x 100m = 10_000 m2
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "polygons_m2.gpkg"

        result = postprocess(
            raster_path,
            output_path,
            threshold=0.5,
            min_area_pixels=1,
            min_area_m2=50_000.0,
        )

        gdf = gpd.read_file(result)
        assert len(gdf) == 0

    def test_empty_raster_has_correct_schema(self, tmp_path: Path) -> None:
        """All-zero raster should produce empty GeoPackage with expected columns."""
        probs = np.zeros((128, 128), dtype=np.float32)
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "empty.gpkg"

        result = postprocess(raster_path, output_path, threshold=0.5)

        gdf = gpd.read_file(result)
        assert len(gdf) == 0
        assert "detection_id" in gdf.columns
        assert "confidence" in gdf.columns
        assert "area_m2" in gdf.columns
        assert "area_ha" in gdf.columns

    def test_partial_min_area_m2_filter(self, tmp_path: Path) -> None:
        """Only polygons >= min_area_m2 should remain with sequential IDs."""
        probs = np.zeros((256, 256), dtype=np.float32)
        # Large square: 100x100 px @ 10m = 10000 m2
        probs[20:120, 20:120] = 0.8
        # Small square: 2x2 px @ 10m = 20m x 20m = 400 m2
        probs[180:182, 180:182] = 0.8
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "polygons_partial.gpkg"

        result = postprocess(
            raster_path,
            output_path,
            threshold=0.5,
            min_area_pixels=1,
            min_area_m2=500.0,
        )

        gdf = gpd.read_file(result)
        assert len(gdf) == 1
        assert gdf.iloc[0]["detection_id"] == 1
        assert gdf.iloc[0]["area_m2"] >= 500.0
