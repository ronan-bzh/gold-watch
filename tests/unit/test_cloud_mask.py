"""Tests for cloud masking and quality filtering."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from goldmine_watch.data.cloud_mask import compute_valid_fraction, create_cloud_mask
from goldmine_watch.data.patches import generate_sliding_window_patches

TARGET_CRS = "EPSG:2972"
RESOLUTION = 10.0
ORIGIN_X = 200_000.0
ORIGIN_Y = 500_000.0
IMAGE_SIZE = 512
NUM_BANDS = 9


def _write_scl_sidecar(image_path: Path, scl_data: np.ndarray) -> Path:
    """Write a single-band SCL sidecar GeoTIFF next to the image."""
    scl_path = image_path.with_suffix("").with_name(image_path.stem + "_SCL").with_suffix(".tif")
    transform = from_origin(ORIGIN_X, ORIGIN_Y + IMAGE_SIZE * RESOLUTION, RESOLUTION, RESOLUTION)
    with rasterio.open(
        scl_path,
        "w",
        driver="GTiff",
        height=scl_data.shape[0],
        width=scl_data.shape[1],
        count=1,
        dtype=scl_data.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        dst.write(scl_data, 1)
        dst.update_tags(1, BAND="SCL")
    return scl_path


class TestCloudMask:
    """Test suite for cloud masking utilities."""

    def test_scl_to_binary_mask(self) -> None:
        """SCL classes [0,3,8,9] should become 0; others become 1."""
        scl = np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [8, 9, 10],
            ],
            dtype=np.uint8,
        )
        mask = create_cloud_mask(scl, invalid_classes=[0, 3, 8, 9])
        expected = np.array(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_all_cloudy_returns_zero_fraction(self) -> None:
        """100%% cloud cover → valid_fraction = 0.0."""
        scl = np.full((10, 10), 9, dtype=np.uint8)  # class 9 = cloud high
        mask = create_cloud_mask(scl)
        assert compute_valid_fraction(mask) == 0.0

    def test_no_clouds_returns_one_fraction(self) -> None:
        """0%% cloud cover → valid_fraction = 1.0."""
        scl = np.full((10, 10), 4, dtype=np.uint8)  # class 4 = vegetation
        mask = create_cloud_mask(scl)
        assert compute_valid_fraction(mask) == 1.0

    def test_patch_rejected_if_too_cloudy(
        self, synthetic_geotiff: Path, synthetic_labels: Path
    ) -> None:
        """Patch with 90%% clouds and threshold 80%% should be rejected."""
        # Create an SCL sidecar where the top-left 256x256 patch is 90%% cloudy
        scl = np.full((IMAGE_SIZE, IMAGE_SIZE), 4, dtype=np.uint8)
        # Set 90%% of the top-left patch pixels to cloudy
        cloudy_pixels = int(0.9 * 256 * 256)
        flat = scl[:256, :256].ravel()
        flat[:cloudy_pixels] = 9
        scl[:256, :256] = flat.reshape(256, 256)
        _write_scl_sidecar(synthetic_geotiff, scl)

        patches = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=10,
            min_valid_fraction=0.8,
        )
        # Top-left patch is ~90%% cloudy → rejected. Other 3 patches are clear → accepted.
        assert len(patches) == 3

    def test_patch_accepted_if_clear(self, synthetic_geotiff: Path, synthetic_labels: Path) -> None:
        """Patch with 10%% clouds and threshold 80%% should be accepted."""
        # Create an SCL sidecar where the top-left 256x256 patch is 10%% cloudy
        scl = np.full((IMAGE_SIZE, IMAGE_SIZE), 4, dtype=np.uint8)
        # Set 10%% of the top-left patch pixels to cloudy
        cloudy_pixels = int(0.1 * 256 * 256)
        flat = scl[:256, :256].ravel()
        flat[:cloudy_pixels] = 9
        scl[:256, :256] = flat.reshape(256, 256)
        _write_scl_sidecar(synthetic_geotiff, scl)

        patches = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=10,
            min_valid_fraction=0.8,
        )
        # All 4 patches are at least 90%% clear → accepted.
        assert len(patches) == 4
