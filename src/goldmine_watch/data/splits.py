"""Spatial train/validation split for geospatial patches."""

import math
import random
from pathlib import Path


def spatial_train_val_split(
    patches_dir: str | Path,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split patches into train/val based on spatial location.

    Groups patches by approximate geographic quadrant to ensure
    train and val come from different areas. Patches are assumed
    to be numbered in raster-scan order (top-left to bottom-right).

    Args:
        patches_dir: Directory containing image_*.npy and mask_*.npy files.
        val_ratio: Approximate fraction of patches to reserve for validation.
        random_seed: Seed for deterministic quadrant assignment.

    Returns:
        Tuple of (train_files, val_files) lists of image Paths.
    """
    patches_dir = Path(patches_dir)
    image_files = sorted(patches_dir.glob("image_*.npy"))
    if not image_files:
        raise ValueError(f"No image_*.npy files found in {patches_dir}")

    n = len(image_files)
    # Estimate grid dimensions assuming roughly square layout.
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    mid_row = rows / 2
    mid_col = cols / 2

    quadrants: dict[int, list[Path]] = {0: [], 1: [], 2: [], 3: []}
    for idx, f in enumerate(image_files):
        row = idx // cols
        col = idx % cols
        q = (0 if col < mid_col else 1) + (0 if row < mid_row else 2)
        quadrants[q].append(f)

    rng = random.Random(random_seed)
    quad_order = list(quadrants.keys())
    rng.shuffle(quad_order)

    val_target = n * val_ratio
    val_files: list[Path] = []
    train_files: list[Path] = []

    for q in quad_order:
        if len(val_files) < val_target:
            val_files.extend(quadrants[q])
        else:
            train_files.extend(quadrants[q])

    return train_files, val_files
