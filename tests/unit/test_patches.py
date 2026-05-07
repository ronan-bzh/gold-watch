"""Tests for patch extraction."""

from pathlib import Path

import numpy as np

from goldmine_watch.data.patches import (
    generate_sliding_window_patches,
    make_patch,
    save_patch_visual,
)


class TestMakePatch:
    """Test single patch extraction."""

    def test_make_patch_shape(self, synthetic_geotiff: Path, synthetic_labels: Path) -> None:
        """Should return image and mask of the requested size."""
        image_patch, mask_patch = make_patch(
            synthetic_geotiff, synthetic_labels, x=0, y=0, size=256
        )
        assert image_patch.shape == (9, 256, 256)
        assert mask_patch.shape == (256, 256)

    def test_make_patch_mask_values(self, synthetic_geotiff: Path, synthetic_labels: Path) -> None:
        """Mask should contain only 0 and 1."""
        _, mask_patch = make_patch(synthetic_geotiff, synthetic_labels, x=0, y=0, size=256)
        assert set(np.unique(mask_patch)).issubset({0, 1})


class TestSlidingWindow:
    """Test sliding window patch generation."""

    def test_generate_patches(self, synthetic_geotiff: Path, synthetic_labels: Path) -> None:
        """Should generate multiple patches up to max_patches."""
        patches = generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=128,
            stride=128,
            max_patches=10,
        )
        assert len(patches) <= 10
        assert len(patches) > 0
        for image_patch, mask_patch in patches:
            assert image_patch.shape[1:] == (128, 128)
            assert mask_patch.shape == (128, 128)

    def test_save_patches_to_disk(
        self, synthetic_geotiff: Path, synthetic_labels: Path, tmp_path: Path
    ) -> None:
        """Should save .npy files when output_dir is provided."""
        out_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=128,
            stride=128,
            max_patches=5,
            output_dir=out_dir,
        )
        assert len(list(out_dir.glob("image_*.npy"))) == 5
        assert len(list(out_dir.glob("mask_*.npy"))) == 5


class TestSavePatchVisual:
    """Test visual patch saving."""

    def test_save_visual(
        self, synthetic_geotiff: Path, synthetic_labels: Path, tmp_path: Path
    ) -> None:
        """Should create a PNG file."""
        image_patch, mask_patch = make_patch(
            synthetic_geotiff, synthetic_labels, x=0, y=0, size=256
        )
        out_path = save_patch_visual(image_patch, mask_patch, tmp_path, prefix="test")
        assert out_path.exists()
        assert out_path.suffix == ".png"
