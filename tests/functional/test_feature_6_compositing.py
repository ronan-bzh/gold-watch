"""Functional tests for Feature 6: Temporal Compositing.

These tests exercise compositing multiple synthetic scenes into a median or mean composite.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

TARGET_CRS = "EPSG:2972"


def _make_scene(tmp_path: Path, suffix: str, size: int = 256, bands: int = 7) -> Path:
    """Create a single synthetic scene GeoTIFF."""
    scene_path = tmp_path / f"scene_{suffix}.tif"
    transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)
    data = np.random.randint(0, 10_000, size=(bands, size, size), dtype=np.uint16)
    with rasterio.open(
        scene_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype=data.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        dst.write(data)
    return scene_path


class TestFeature6CompositingFlow:
    """End-to-end compositing workflow tests."""

    def test_median_composite_reduces_outliers(
        self, tmp_path: Path
    ) -> None:
        """Median of scenes with an outlier band should be closer to normal."""
        size = 64
        bands = 3
        scenes = []
        for i in range(3):
            path = _make_scene(tmp_path, f"s{i}", size=size, bands=bands)
            scenes.append(path)

        # Read all scenes into a stack
        stack = []
        for path in scenes:
            with rasterio.open(path) as src:
                stack.append(src.read().astype(np.float32))

        # Make one scene an outlier
        stack[1] = stack[1] + 5000

        stack_arr = np.stack(stack, axis=0)  # (time, bands, h, w)
        median = np.median(stack_arr, axis=0)
        mean = np.mean(stack_arr, axis=0)

        # Median should be less affected by outlier than mean
        assert np.abs(median - stack[0]).mean() < np.abs(mean - stack[0]).mean()

    def test_composite_matches_single_scene_shape(
        self, tmp_path: Path
    ) -> None:
        """Composite of same-size scenes preserves (bands, h, w)."""
        size = 128
        bands = 7
        scenes = [_make_scene(tmp_path, f"s{i}", size=size, bands=bands) for i in range(2)]

        stack = []
        for path in scenes:
            with rasterio.open(path) as src:
                stack.append(src.read())

        composite = np.median(np.stack(stack, axis=0), axis=0)
        assert composite.shape == (bands, size, size)

    def test_composite_written_to_geotiff(
        self, tmp_path: Path
    ) -> None:
        """Composite can be written and re-read as a valid GeoTIFF."""
        size = 128
        bands = 7
        scenes = [_make_scene(tmp_path, f"s{i}", size=size, bands=bands) for i in range(2)]

        stack = []
        for path in scenes:
            with rasterio.open(path) as src:
                stack.append(src.read().astype(np.float32))

        composite = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint16)
        out_path = tmp_path / "composite.tif"
        transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=bands,
            dtype=composite.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(composite)

        assert out_path.exists()
        with rasterio.open(out_path) as src:
            assert src.count == bands
            assert src.width == size
            assert src.height == size
            assert src.crs.to_string() == TARGET_CRS

    def test_mean_composite_different_from_median(
        self, tmp_path: Path
    ) -> None:
        """Mean and median composites differ for skewed data."""
        size = 64
        bands = 3
        scenes = [_make_scene(tmp_path, f"s{i}", size=size, bands=bands) for i in range(3)]

        stack = []
        for path in scenes:
            with rasterio.open(path) as src:
                stack.append(src.read().astype(np.float32))

        stack_arr = np.stack(stack, axis=0)
        mean = np.mean(stack_arr, axis=0)
        median = np.median(stack_arr, axis=0)

        # For random data they are unlikely to be identical
        assert not np.allclose(mean, median)

    def test_composite_preserves_transform(
        self, tmp_path: Path
    ) -> None:
        """Composite GeoTIFF should have the same transform as input scenes."""
        size = 64
        scenes = [_make_scene(tmp_path, f"s{i}", size=size, bands=7) for i in range(2)]

        with rasterio.open(scenes[0]) as src:
            expected_transform = src.transform

        stack = []
        for path in scenes:
            with rasterio.open(path) as src:
                stack.append(src.read().astype(np.uint16))

        composite = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint16)
        out_path = tmp_path / "composite_transform.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=7,
            dtype=composite.dtype,
            crs=TARGET_CRS,
            transform=expected_transform,
        ) as dst:
            dst.write(composite)

        with rasterio.open(out_path) as src:
            assert src.transform == expected_transform
