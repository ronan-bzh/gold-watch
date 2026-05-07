"""Generate 5-10 patches from a synthetic image for visual verification.

Usage:
    python scripts/make_patches.py <image.tif> <labels.gpkg>
"""

import random
import sys
from pathlib import Path

from goldmine_watch.data.patches import make_patch, save_patch_visual


def generate_patches(image_path: str | Path, labels_path: str | Path, num_patches: int = 8) -> None:
    """Generate and save random patches for visual inspection."""
    import rasterio

    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height

    patch_size = 256
    output_dir = Path("outputs/patches")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_patches):
        x = random.randint(0, max(0, width - patch_size))
        y = random.randint(0, max(0, height - patch_size))

        image_patch, mask_patch = make_patch(image_path, labels_path, x, y, patch_size)
        save_patch_visual(image_patch, mask_patch, output_dir, prefix=f"patch_{i:02d}_{x}_{y}")
        print(f"Saved patch {i + 1}/{num_patches} at ({x}, {y})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/make_patches.py <image.tif> <labels.gpkg>")
        sys.exit(1)

    generate_patches(sys.argv[1], sys.argv[2])
