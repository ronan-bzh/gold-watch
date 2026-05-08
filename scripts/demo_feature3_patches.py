"""Demo script for Feature 3: Real Patch Generation.

Usage:
    python scripts/demo_feature3_patches.py \
        data/raw/sentinel2_scene.tif \
        data/raw/mining_surfaces.gpkg \
        --num-display 9
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from goldmine_watch.data.patches import generate_sliding_window_patches

if TYPE_CHECKING:
    import matplotlib.figure


def _create_patch_grid(
    patches: list[tuple[np.ndarray, np.ndarray]],
    num_display: int = 9,
) -> matplotlib.figure.Figure:
    """Create a 3×3 grid of patches with mask overlay in red.

    Returns:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    num_display = min(num_display, len(patches))
    selected = random.sample(range(len(patches)), num_display)

    for idx, ax in enumerate(axes):
        if idx < num_display:
            i = selected[idx]
            image_patch, mask_patch = patches[i]

            # Simple RGB composite from first 3 bands, percentile stretch
            rgb = image_patch[:3].transpose(1, 2, 0).astype(np.float32)
            for b in range(3):
                band = rgb[:, :, b]
                p2, p98 = np.percentile(band, (2, 98))
                rgb[:, :, b] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

            # Overlay mask in red
            overlay = rgb.copy()
            overlay[mask_patch > 0] = [1.0, 0.0, 0.0]

            ax.imshow(overlay)
            ax.set_title(f"Patch {i}")
        ax.axis("off")

    plt.tight_layout()
    return fig


def main() -> int:
    """Generate patches and display a grid."""
    parser = argparse.ArgumentParser(
        description="Generate training patches from a satellite image and labels."
    )
    parser.add_argument("image_path", type=Path, help="Path to the GeoTIFF image")
    parser.add_argument("labels_path", type=Path, help="Path to the vector labels")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size in pixels (default: 256)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Sliding window stride (default: 256)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=500,
        help="Maximum number of patches to generate (default: 500)",
    )
    parser.add_argument(
        "--num-display",
        type=int,
        default=9,
        help="Number of patches to show in the grid (default: 9)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/patches"),
        help="Directory to save .npy patches (default: outputs/patches)",
    )
    parser.add_argument(
        "--cloud-mask",
        type=Path,
        default=None,
        help="Optional path to a binary cloud mask .npy file",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"Error: Image not found: {args.image_path}", file=sys.stderr)
        return 1

    if not args.labels_path.exists():
        print(f"Error: Labels not found: {args.labels_path}", file=sys.stderr)
        return 1

    cloud_mask = None
    if args.cloud_mask is not None:
        if not args.cloud_mask.exists():
            print(
                f"Error: Cloud mask not found: {args.cloud_mask}",
                file=sys.stderr,
            )
            return 1
        cloud_mask = np.load(args.cloud_mask)

    print(f"Generating patches from {args.image_path.name} ...")

    result = generate_sliding_window_patches(
        args.image_path,
        args.labels_path,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches=args.max_patches,
        output_dir=args.output_dir,
        cloud_mask=cloud_mask,
        return_stats=True,
    )

    patches = result["patches"]
    rejected = result["rejected"]
    generated = result["generated"]

    print(f"Generated {generated} patches")
    if rejected > 0:
        print(f"Rejected {rejected} patches")
    print(f"Saved to {args.output_dir}")

    if len(patches) == 0:
        print("Warning: no patches were generated.")
        return 0

    # Save visual grid
    demo_dir = Path("outputs/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    grid_path = demo_dir / "patch_grid.png"

    fig = _create_patch_grid(patches, num_display=args.num_display)
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"Saved patch grid to {grid_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
