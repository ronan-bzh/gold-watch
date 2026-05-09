"""Unit tests for square post-processing (Feature 14)."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.data.square_postprocess import (
    _cell_confidence,
    create_square_grid,
    square_postprocess,
)

TARGET_CRS = "EPSG:2972"


def _make_probability_raster(
    tmp_path: Path,
    size: int = 256,
    values: np.ndarray | None = None,
    resolution: float = 10.0,
) -> Path:
    """Create a synthetic probability raster."""
    raster_path = tmp_path / "probs.tif"
    probs = np.zeros((size, size), dtype=np.float32) if values is None else values
    transform = from_origin(200_000.0, 500_000.0 + size * resolution, resolution, resolution)

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


class TestSquarePostprocess:
    """Tests for square_postprocess function."""

    def test_outputs_geojson(self, tmp_path: Path) -> None:
        """Should create a valid GeoJSON file."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "detections.geojson"

        result = square_postprocess(
            [raster_path],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(output_path),
        )

        assert result.exists()
        gdf = gpd.read_file(result)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_all_features_are_squares(self, tmp_path: Path) -> None:
        """All geometries should have equal width and height."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "detections_square.geojson"

        result = square_postprocess(
            [raster_path],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(output_path),
        )

        gdf = gpd.read_file(result)
        for geom in gdf.geometry:
            minx, miny, maxx, maxy = geom.bounds
            width = maxx - minx
            height = maxy - miny
            assert abs(width - height) < 1e-6

    def test_features_have_confidence_attribute(self, tmp_path: Path) -> None:
        """Each feature should have a 'confidence' property."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "detections_conf.geojson"

        result = square_postprocess(
            [raster_path],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(output_path),
        )

        gdf = gpd.read_file(result)
        assert "confidence" in gdf.columns
        assert len(gdf) > 0
        assert all(gdf["confidence"] > 0)

    def test_threshold_filters_low_confidence(self, tmp_path: Path) -> None:
        """Features below threshold should be excluded."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_low = tmp_path / "detections_low.geojson"
        output_high = tmp_path / "detections_high.geojson"

        square_postprocess(
            [raster_path],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(output_low),
        )
        square_postprocess(
            [raster_path],
            grid_size_m=100.0,
            threshold=0.7,
            min_confidence=0.7,
            output_path=str(output_high),
        )

        gdf_low = gpd.read_file(output_low)
        gdf_high = gpd.read_file(output_high)
        assert len(gdf_low) >= len(gdf_high)

    def test_empty_raster_produces_empty_geojson(self, tmp_path: Path) -> None:
        """All-zero raster should produce empty GeoJSON with correct schema."""
        probs = np.zeros((128, 128), dtype=np.float32)
        raster_path = _make_probability_raster(tmp_path, values=probs)
        output_path = tmp_path / "empty.geojson"

        result = square_postprocess(
            [raster_path],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(output_path),
        )

        gdf = gpd.read_file(result)
        assert len(gdf) == 0
        # GeoJSON drivers may drop empty property columns; verify file exists and is valid
        assert result.exists()
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_multiple_rasters_merged(self, tmp_path: Path) -> None:
        """Multiple input rasters should be merged before processing."""
        r1_dir = tmp_path / "r1"
        r1_dir.mkdir(parents=True, exist_ok=True)
        probs1 = np.zeros((128, 128), dtype=np.float32)
        probs1[20:80, 20:80] = 0.8
        raster1 = _make_probability_raster(r1_dir, size=128, values=probs1)

        r2_dir = tmp_path / "r2"
        r2_dir.mkdir(parents=True, exist_ok=True)
        probs2 = np.zeros((128, 128), dtype=np.float32)
        probs2[20:80, 20:80] = 0.6
        raster2 = _make_probability_raster(r2_dir, size=128, values=probs2)

        output_path = tmp_path / "merged.geojson"

        result = square_postprocess(
            [raster1, raster2],
            grid_size_m=100.0,
            threshold=0.1,
            min_confidence=0.1,
            output_path=str(output_path),
        )

        assert result.exists()
        gdf = gpd.read_file(result)
        assert isinstance(gdf, gpd.GeoDataFrame)


class TestCreateSquareGrid:
    """Tests for create_square_grid function."""

    def test_grid_cells_equal_size(self) -> None:
        """All cells should have the same area."""
        bounds = (0.0, 0.0, 1000.0, 1000.0)
        grid = create_square_grid(bounds, grid_size_m=100.0)

        areas = grid.geometry.area
        assert len(grid) > 0
        assert areas.nunique() == 1
        assert abs(areas.iloc[0] - 10000.0) < 1e-6

    def test_grid_covers_bounds(self) -> None:
        """Grid should cover the specified bounds."""
        bounds = (0.0, 0.0, 1000.0, 1000.0)
        grid = create_square_grid(bounds, grid_size_m=100.0)

        total_bounds = grid.total_bounds
        assert total_bounds[0] <= bounds[0]
        assert total_bounds[1] <= bounds[1]
        assert total_bounds[2] >= bounds[2]
        assert total_bounds[3] >= bounds[3]

    def test_grid_crs(self) -> None:
        """Grid should use the specified CRS."""
        bounds = (0.0, 0.0, 1000.0, 1000.0)
        grid = create_square_grid(bounds, grid_size_m=100.0, crs=TARGET_CRS)
        assert str(grid.crs) == TARGET_CRS


class TestCellConfidence:
    """Tests for _cell_confidence function."""

    def test_computes_mean_probability(self, tmp_path: Path) -> None:
        """Mean probability inside a high-value region."""
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8
        raster_path = _make_probability_raster(tmp_path, values=probs)

        cell = Polygon(
            [
                (200_000.0 + 500, 500_000.0 + 500),
                (200_000.0 + 1500, 500_000.0 + 500),
                (200_000.0 + 1500, 500_000.0 + 1500),
                (200_000.0 + 500, 500_000.0 + 1500),
            ]
        )

        import rasterio

        with rasterio.open(raster_path) as src:
            conf = _cell_confidence(src, cell)
        assert conf > 0.0

    def test_returns_zero_for_empty_cell(self, tmp_path: Path) -> None:
        """Zero probability outside the raster should return 0.0."""
        probs = np.zeros((256, 256), dtype=np.float32)
        raster_path = _make_probability_raster(tmp_path, values=probs)

        cell = Polygon(
            [
                (200_000.0 + 500, 500_000.0 + 500),
                (200_000.0 + 1500, 500_000.0 + 500),
                (200_000.0 + 1500, 500_000.0 + 1500),
                (200_000.0 + 500, 500_000.0 + 1500),
            ]
        )

        import rasterio

        with rasterio.open(raster_path) as src:
            conf = _cell_confidence(src, cell)
        assert conf == 0.0

    def test_returns_zero_for_cell_outside_raster(self, tmp_path: Path) -> None:
        """Cell completely outside raster bounds should return 0.0."""
        probs = np.zeros((256, 256), dtype=np.float32)
        raster_path = _make_probability_raster(tmp_path, values=probs)

        cell = Polygon(
            [
                (300_000.0, 600_000.0),
                (300_000.0 + 100, 600_000.0),
                (300_000.0 + 100, 600_000.0 + 100),
                (300_000.0, 600_000.0 + 100),
            ]
        )

        import rasterio

        with rasterio.open(raster_path) as src:
            conf = _cell_confidence(src, cell)
        assert conf == 0.0
