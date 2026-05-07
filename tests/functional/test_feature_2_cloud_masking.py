"""Functional tests for Feature 2: Cloud Masking & Quality Filtering.

These tests exercise the full cloud-mask workflow integrated with
patch generation, using synthetic SCL data.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from goldmine_watch.data.cloud_mask import (
    apply_cloud_mask,
    compute_valid_fraction,
    create_cloud_mask,
    load_scl_band,
)
from goldmine_watch.data.patches import generate_sliding_window_patches

TARGET_CRS = "EPSG:2972"
RESOLUTION = 10.0
ORIGIN_X = 200_000.0
ORIGIN_Y = 500_000.0
IMAGE_SIZE = 512
NUM_BANDS = 7


def _write_scl_sidecar(image_path: Path, scl_data: np.ndarray) -> Path:
    """Write a single-band SCL sidecar GeoTIFF next to the image."""
    scl_path = (
        image_path.with_suffix("").with_name(image_path.stem + "_SCL").with_suffix(".tif")
    )
    transform = from_origin(
        ORIGIN_X, ORIGIN_Y + IMAGE_SIZE * RESOLUTION, RESOLUTION, RESOLUTION
    )
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


class TestFeature2CloudMaskFlow:
    """End-to-end cloud masking workflow tests."""

    def test_load_scl_and_create_mask(
        self, synthetic_geotiff: Path
    ) -> None:
        """Load SCL from image tag and produce a correct binary mask."""
        # The synthetic fixture tags band 7 as SCL
        scl = load_scl_band(synthetic_geotiff)
        assert scl.shape == (IMAGE_SIZE, IMAGE_SIZE)

        mask = create_cloud_mask(scl, invalid_classes=[0, 3, 8, 9])
        assert mask.shape == scl.shape
        assert set(np.unique(mask)).issubset({0, 1})

    def test_valid_fraction_all_clear(self) -> None:
        """100%% clear pixels -> valid_fraction = 1.0."""
        scl = np.full((10, 10), 4, dtype=np.uint8)  # vegetation = valid
        mask = create_cloud_mask(scl)
        assert compute_valid_fraction(mask) == 1.0

    def test_valid_fraction_all_cloudy(self) -> None:
        """100%% cloudy pixels -> valid_fraction = 0.0."""
        scl = np.full((10, 10), 9, dtype=np.uint8)  # cloud high = invalid
        mask = create_cloud_mask(scl)
        assert compute_valid_fraction(mask) == 0.0

    def test_patches_filtered_by_cloud_mask(
        self, synthetic_geotiff: Path, synthetic_labels: Path
    ) -> None:
        """Cloudy patches are rejected when cloud_mask is supplied."""
        # Create a cloud mask where top-left 256x256 is fully cloudy
        cloud_mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cloud_mask[:256, :256] = 0

        patches = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=10,
            max_cloud_fraction=0.2,
            cloud_mask=cloud_mask,
        )
        # 512x512 with stride 256 yields 4 patches; top-left is rejected
        assert len(patches) == 3

    def test_custom_invalid_classes_change_mask(
        self, tmp_path: Path
    ) -> None:
        """Custom SCL invalid classes change which pixels are masked."""
        scl = np.array([[0, 1, 2], [3, 4, 5], [8, 9, 10]], dtype=np.uint8)

        default_mask = create_cloud_mask(scl, invalid_classes=[0, 3, 8, 9])
        custom_mask = create_cloud_mask(scl, invalid_classes=[0, 8])

        # Default should mask more pixels
        assert default_mask.sum() < custom_mask.sum()
        np.testing.assert_array_equal(
            default_mask,
            np.array([[0, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            custom_mask,
            np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=np.uint8),
        )

    def test_apply_cloud_mask_zeroes_invalid_pixels(self) -> None:
        """apply_cloud_mask sets invalid pixels to 0 across all bands."""
        image = np.ones((3, 4, 4), dtype=np.uint16) * 100
        mask = np.ones((4, 4), dtype=np.uint8)
        mask[0, 0] = 0
        mask[1, 1] = 0

        masked = apply_cloud_mask(image, mask)
        assert masked[0, 0, 0] == 0
        assert masked[1, 1, 1] == 0
        assert masked[2, 2, 2] == 100
