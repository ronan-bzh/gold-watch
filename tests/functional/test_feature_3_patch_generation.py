"""Functional tests for Feature 3: Real Patch Generation.

These tests exercise the full patch generation workflow including
sliding-window extraction, cloud filtering, disk output, and stats.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.data.patches import generate_sliding_window_patches, make_patch

TARGET_CRS = "EPSG:2972"


class TestFeature3PatchGenerationFlow:
    """End-to-end patch generation workflow tests."""

    def test_generate_and_save_patches(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Patches are written to disk as paired .npy files."""
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

        # Verify saved files load back correctly
        img_file = sorted(out_dir.glob("image_*.npy"))[0]
        mask_file = sorted(out_dir.glob("mask_*.npy"))[0]
        loaded_img = np.load(img_file)
        loaded_mask = np.load(mask_file)
        assert loaded_img.shape == (7, 256, 256)
        assert loaded_mask.shape == (256, 256)

    def test_patch_shapes_match_config(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
    ) -> None:
        """All generated patches match the expected band and spatial dims."""
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
            assert set(np.unique(mask_patch)).issubset({0, 1})

    def test_stride_doubles_patch_count(
        self, tmp_path: Path
    ) -> None:
        """stride=128 produces roughly 4x more patches than stride=256."""
        image_path = tmp_path / "large_image.tif"
        labels_path = tmp_path / "large_labels.gpkg"

        size = 1024
        transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(7, size, size), dtype=np.uint16)

        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=7,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)

        polygon = Polygon([
            (200_000.0 + 100, 500_000.0 + 100),
            (200_000.0 + 200, 500_000.0 + 100),
            (200_000.0 + 200, 500_000.0 + 200),
            (200_000.0 + 100, 500_000.0 + 200),
        ])
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
        assert 2.0 < ratio < 5.0

    def test_stats_reported_correctly(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
    ) -> None:
        """return_stats=True yields accurate generated and rejected counts."""
        # Create a cloud mask that rejects exactly one patch
        cloud_mask = np.ones((512, 512), dtype=np.uint8)
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

    def test_stats_without_rejections(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
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

    def test_mask_aligns_with_labels(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
    ) -> None:
        """Burned mask contains positive pixels where labels exist."""
        image_patch, mask_patch = make_patch(
            synthetic_geotiff, synthetic_labels, x=0, y=0, size=256
        )
        # First synthetic label sits at pixel coords ~50-150 in both axes
        assert mask_patch.sum() > 0
        assert set(np.unique(mask_patch)).issubset({0, 1})
