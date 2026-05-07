"""Tests for real patch generation with cloud filtering and validation."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.data.patches import (
    _compute_patch_cloud_fraction,
    _compute_patch_valid_fraction,
    generate_sliding_window_patches,
    make_patch,
)

TARGET_CRS = "EPSG:2972"


class TestRealPatchGeneration:
    """Test real patch generation with filtering."""

    def test_patches_have_correct_shape(
        self, synthetic_geotiff: Path, synthetic_labels: Path
    ) -> None:
        """All patches should be (7, 256, 256) and (256, 256)."""
        patches = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
        )
        assert len(patches) > 0
        for image_patch, mask_patch in patches:
            assert image_patch.shape == (7, 256, 256)
            assert mask_patch.shape == (256, 256)

    def test_patches_saved_to_disk(
        self, synthetic_geotiff: Path, synthetic_labels: Path, tmp_path: Path
    ) -> None:
        """output_dir provided → .npy files exist."""
        out_dir = tmp_path / "patches"
        patches = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=out_dir,
        )
        assert len(patches) > 0
        assert len(list(out_dir.glob("image_*.npy"))) == len(patches)
        assert len(list(out_dir.glob("mask_*.npy"))) == len(patches)

    def test_cloudy_patch_rejected(self, synthetic_geotiff: Path, synthetic_labels: Path) -> None:
        """Patch with cloud_fraction > max_cloud_fraction is skipped."""
        with rasterio.open(synthetic_geotiff) as src:
            height = src.height
            width = src.width

        # Top-left quadrant fully cloudy (0), rest clear (1)
        cloud_mask = np.ones((height, width), dtype=np.uint8)
        cloud_mask[:256, :256] = 0

        patches = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            max_cloud_fraction=0.2,
            cloud_mask=cloud_mask,
        )

        # Without cloud filtering a 512×512 image yields 4 patches.
        # One quadrant is cloudy → 3 patches remain.
        assert len(patches) == 3

    def test_mask_aligns_with_labels(self, synthetic_geotiff: Path, synthetic_labels: Path) -> None:
        """Burned mask should have 1s where labels exist."""
        image_patch, mask_patch = make_patch(
            synthetic_geotiff, synthetic_labels, x=0, y=0, size=256
        )
        # First synthetic label sits at pixel coords ~50–150 in both axes,
        # so this patch must contain some positive mask pixels.
        assert mask_patch.sum() > 0
        assert set(np.unique(mask_patch)).issubset({0, 1})

    def test_stride_controls_overlap(self, tmp_path: Path) -> None:
        """stride=128 should produce ~4× more patches than stride=256."""
        image_path = tmp_path / "large_synthetic.tif"
        labels_path = tmp_path / "large_labels.gpkg"

        size = 1024
        transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(9, size, size), dtype=np.uint16)

        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=9,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)

        polygon = Polygon(
            [
                (200_000.0 + 100, 500_000.0 + 100),
                (200_000.0 + 200, 500_000.0 + 100),
                (200_000.0 + 200, 500_000.0 + 200),
                (200_000.0 + 100, 500_000.0 + 200),
            ]
        )
        gdf = gpd.GeoDataFrame(
            {"label": ["mining"]},
            geometry=[polygon],
            crs=TARGET_CRS,
        )
        gdf.to_file(labels_path, driver="GPKG")

        patches_256 = generate_sliding_window_patches(
            image_path,
            labels_path,
            patch_size=256,
            stride=256,
            max_patches=500,
        )
        patches_128 = generate_sliding_window_patches(
            image_path,
            labels_path,
            patch_size=256,
            stride=128,
            max_patches=500,
        )

        ratio = len(patches_128) / len(patches_256)
        # For a 1024×1024 image the theoretical asymptotic ratio is 4.
        # The actual ratio is ~3.1; accept anything reasonably close.
        assert 2.0 < ratio < 5.0


class TestPatchHelpers:
    """Test internal helper functions."""

    def test_compute_patch_cloud_fraction_all_clear(self) -> None:
        """All valid pixels → cloud fraction 0.0."""
        mask = np.ones((10, 10), dtype=np.uint8)
        assert _compute_patch_cloud_fraction(mask) == 0.0

    def test_compute_patch_cloud_fraction_all_cloudy(self) -> None:
        """All cloudy pixels → cloud fraction 1.0."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert _compute_patch_cloud_fraction(mask) == 1.0

    def test_compute_patch_valid_fraction_no_nodata(self) -> None:
        """No nodata value → all pixels valid (1.0)."""
        patch = np.random.rand(3, 10, 10).astype(np.float32)
        assert _compute_patch_valid_fraction(patch, nodata=None) == 1.0

    def test_compute_patch_valid_fraction_with_nodata(self) -> None:
        """Pixels equal to nodata are invalid."""
        patch = np.ones((3, 10, 10), dtype=np.float32)
        patch[:, 0, 0] = 0.0
        patch[:, 1, 1] = 0.0
        # Two pixels out of 100 are nodata in all bands
        expected = 1.0 - (2 / 100)
        assert _compute_patch_valid_fraction(patch, nodata=0.0) == pytest.approx(expected)

    def test_compute_patch_valid_fraction_with_nan_nodata(self) -> None:
        """NaN nodata values are handled correctly."""
        patch = np.ones((3, 10, 10), dtype=np.float32)
        patch[:, 0, 0] = np.nan
        patch[:, 1, 1] = np.nan
        # Two pixels out of 100 are NaN in all bands
        expected = 1.0 - (2 / 100)
        assert _compute_patch_valid_fraction(patch, nodata=np.nan) == pytest.approx(expected)


class TestReturnStats:
    """Test return_stats=True code path."""

    def test_return_stats_with_rejections(
        self, synthetic_geotiff: Path, synthetic_labels: Path
    ) -> None:
        """return_stats=True reports correct rejected and generated counts."""
        with rasterio.open(synthetic_geotiff) as src:
            height = src.height
            width = src.width

        cloud_mask = np.ones((height, width), dtype=np.uint8)
        cloud_mask[:256, :256] = 0

        stats = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            max_cloud_fraction=0.2,
            cloud_mask=cloud_mask,
            return_stats=True,
        )

        assert isinstance(stats, dict)
        assert stats["rejected"] == 1
        assert stats["generated"] == 3
        assert len(stats["patches"]) == 3

    def test_return_stats_without_rejections(
        self, synthetic_geotiff: Path, synthetic_labels: Path
    ) -> None:
        """return_stats=True reports zero rejected when nothing is filtered."""
        stats = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            return_stats=True,
        )

        assert isinstance(stats, dict)
        assert stats["rejected"] == 0
        assert stats["generated"] == 4
        assert len(stats["patches"]) == 4


class TestCloudMaskValidation:
    """Test cloud mask shape validation."""

    def test_wrong_shape_cloud_mask_raises(
        self, synthetic_geotiff: Path, synthetic_labels: Path
    ) -> None:
        """A cloud mask with mismatched dimensions raises ValueError."""
        with rasterio.open(synthetic_geotiff) as src:
            height = src.height
            width = src.width

        # Intentionally wrong shape
        cloud_mask = np.ones((height + 10, width - 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="cloud_mask shape"):
            generate_sliding_window_patches(
                synthetic_geotiff,
                synthetic_labels,
                patch_size=256,
                stride=256,
                cloud_mask=cloud_mask,
            )
