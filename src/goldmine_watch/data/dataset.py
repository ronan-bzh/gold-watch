"""PyTorch Dataset for loading saved patches from disk."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Dataset that loads image/mask .npy pairs from a directory."""

    def __init__(self, patches_dir: str | Path, augment: bool = True) -> None:
        """Initialize the dataset.

        Args:
            patches_dir: Directory containing image_*.npy and mask_*.npy files.
            augment: If True, apply random horizontal flips.
        """
        self.patches_dir = Path(patches_dir)
        self.augment = augment
        self.image_files = sorted(self.patches_dir.glob("image_*.npy"))
        if not self.image_files:
            raise ValueError(f"No image_*.npy files found in {patches_dir}")

    def __len__(self) -> int:
        """Return the number of patch pairs."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return the image and mask at the given index."""
        image_path = self.image_files[idx]
        mask_path = self.patches_dir / image_path.name.replace("image_", "mask_")

        image = np.load(image_path).astype(np.float32)  # (C, H, W)
        mask = np.load(mask_path).astype(np.float32)  # (H, W)

        if self.augment and np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        image_t = torch.from_numpy(image)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return image_t, mask_t
