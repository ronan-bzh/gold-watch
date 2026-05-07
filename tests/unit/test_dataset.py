"""Tests for the patch dataset."""

from pathlib import Path

import numpy as np
import pytest
import torch

from goldmine_watch.data.dataset import PatchDataset


class TestPatchDataset:
    """Test suite for PatchDataset."""

    @pytest.fixture
    def patches_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory with a few fake patch pairs."""
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()
        for i in range(3):
            image = np.random.randint(0, 10_000, size=(9, 256, 256), dtype=np.uint16)
            mask = np.random.randint(0, 2, size=(256, 256), dtype=np.uint8)
            np.save(patches_dir / f"image_{i:04d}.npy", image)
            np.save(patches_dir / f"mask_{i:04d}.npy", mask)
        return patches_dir

    def test_loads_all_patches(self, patches_dir: Path) -> None:
        """Should load all patch pairs."""
        dataset = PatchDataset(patches_dir, augment=False)
        assert len(dataset) == 3

    def test_returns_tensors(self, patches_dir: Path) -> None:
        """Should return torch Tensors with correct shapes."""
        dataset = PatchDataset(patches_dir, augment=False)
        image, mask = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (9, 256, 256)
        assert mask.shape == (1, 256, 256)

    def test_augmentation_does_not_crash(self, patches_dir: Path) -> None:
        """Should apply augmentation without errors."""
        dataset = PatchDataset(patches_dir, augment=True)
        image, mask = dataset[0]
        assert image.shape == (9, 256, 256)
        assert mask.shape == (1, 256, 256)
