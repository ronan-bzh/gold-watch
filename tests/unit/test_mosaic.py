"""Unit tests for mosaic builder (Feature 15)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from goldmine_watch.data.mosaic import build_mosaic, get_mosaic_bounds, validate_mosaic

TARGET_CRS = "EPSG:2972"


def _make_probability_raster(
    tmp_path: Path,
    size: int = 128,
    values: np.ndarray | None = None,
    origin_x: float = 200_000.0,
    origin_y: float = 500_000.0,
    resolution: float = 10.0,
) -> Path:
    """Create a synthetic single-band probability GeoTIFF."""
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


class TestBuildMosaic:
    """Tests for build_mosaic function."""

    def test_mosaic_created(self, tmp_path: Path) -> None:
        """Should create a valid GeoTIFF."""
        probs = np.zeros((128, 128), dtype=np.float32)
        probs[20:108, 20:108] = 0.5
        raster_path = _make_probability_raster(tmp_path, values=probs)

        output_path = tmp_path / "mosaic.tif"
        result = build_mosaic([raster_path], output_path=str(output_path), method="mean")

        assert result.exists()
        with rasterio.open(result) as src:
            assert src.count == 1
            assert src.width > 0
            assert src.height > 0

    def test_mosaic_covers_all_inputs(self, tmp_path: Path) -> None:
        """Mosaic bounds should encompass all input bounds."""
        r1 = _make_probability_raster(
            tmp_path, size=64, origin_x=200_000.0, origin_y=500_000.0
        )
        r2 = _make_probability_raster(
            tmp_path, size=64, origin_x=200_640.0, origin_y=500_000.0
        )

        output_path = tmp_path / "mosaic.tif"
        result = build_mosaic([r1, r2], output_path=str(output_path), method="mean")

        with rasterio.open(result) as src:
            mosaic_bounds = src.bounds

        union = get_mosaic_bounds([r1, r2])

        assert pytest.approx(mosaic_bounds.left, abs=1e-3) == union[0]
        assert pytest.approx(mosaic_bounds.bottom, abs=1e-3) == union[1]
        assert pytest.approx(mosaic_bounds.right, abs=1e-3) == union[2]
        assert pytest.approx(mosaic_bounds.top, abs=1e-3) == union[3]

    def test_no_gaps_in_mosaic(self, tmp_path: Path) -> None:
        """Mosaic should have no nodata holes."""
        probs1 = np.ones((128, 128), dtype=np.float32) * 0.3
        probs2 = np.ones((128, 128), dtype=np.float32) * 0.7
        r1 = _make_probability_raster(tmp_path, values=probs1, origin_x=200_000.0)
        r2 = _make_probability_raster(
            tmp_path, values=probs2, origin_x=200_000.0, origin_y=501_280.0
        )

        output_path = tmp_path / "mosaic.tif"
        result = build_mosaic([r1, r2], output_path=str(output_path), method="mean")

        report = validate_mosaic(result)
        assert not report["has_gaps"]

    def test_max_method_takes_maximum(self, tmp_path: Path) -> None:
        """Max method should pick the highest value in overlaps."""
        probs1 = np.ones((128, 128), dtype=np.float32) * 0.3
        probs2 = np.ones((128, 128), dtype=np.float32) * 0.7
        r1 = _make_probability_raster(tmp_path, values=probs1, origin_x=200_000.0)
        r2 = _make_probability_raster(tmp_path, values=probs2, origin_x=200_000.0)

        output_path = tmp_path / "mosaic_max.tif"
        result = build_mosaic([r1, r2], output_path=str(output_path), method="max")

        with rasterio.open(result) as src:
            data = src.read(1, masked=True)
            valid = data.compressed()
            assert valid.max() == pytest.approx(0.7, abs=1e-5)


class TestGetMosaicBounds:
    """Tests for get_mosaic_bounds function."""

    def test_computes_union_of_bounds(self, tmp_path: Path) -> None:
        """Should return min of mins and max of maxes."""
        r1 = _make_probability_raster(
            tmp_path, size=64, origin_x=200_000.0, origin_y=500_000.0
        )
        r2 = _make_probability_raster(
            tmp_path, size=64, origin_x=200_640.0, origin_y=500_640.0
        )

        bounds = get_mosaic_bounds([r1, r2])

        assert bounds[0] == pytest.approx(200_000.0, abs=1e-3)
        assert bounds[1] == pytest.approx(500_000.0, abs=1e-3)
        assert bounds[2] == pytest.approx(201_280.0, abs=1e-3)
        assert bounds[3] == pytest.approx(501_280.0, abs=1e-3)

    def test_empty_list_raises(self) -> None:
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            get_mosaic_bounds([])
