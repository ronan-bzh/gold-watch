"""Unit tests for spatial train/val split."""

from pathlib import Path

import numpy as np
import pytest

from goldmine_watch.data.splits import spatial_train_val_split


class TestSpatialSplit:
    """Tests for spatial train/val split."""

    def test_no_overlap_between_splits(self, tmp_path: Path) -> None:
        """Train and val should have no patches in common."""
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()
        for i in range(20):
            np.save(patches_dir / f"image_{i:04d}.npy", np.zeros((3, 16, 16)))
            np.save(patches_dir / f"mask_{i:04d}.npy", np.zeros((16, 16)))

        train_files, val_files = spatial_train_val_split(patches_dir, val_ratio=0.2)
        train_set = {f.name for f in train_files}
        val_set = {f.name for f in val_files}
        assert len(train_set & val_set) == 0

    def test_split_ratio_approximate(self, tmp_path: Path) -> None:
        """val_ratio=0.2 → val set ≈ 20% of total."""
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()
        for i in range(20):
            np.save(patches_dir / f"image_{i:04d}.npy", np.zeros((3, 16, 16)))
            np.save(patches_dir / f"mask_{i:04d}.npy", np.zeros((16, 16)))

        train_files, val_files = spatial_train_val_split(patches_dir, val_ratio=0.2)
        total = len(train_files) + len(val_files)
        val_ratio_actual = len(val_files) / total
        assert val_ratio_actual == pytest.approx(0.2, abs=0.15)
