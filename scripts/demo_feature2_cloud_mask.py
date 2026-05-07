#!/usr/bin/env python3
"""Demo script for Feature 2: Cloud Masking & Quality Filtering.

Usage:
    python scripts/demo_feature2_cloud_mask.py data/raw/sentinel2_scene.tif

Outputs three PNGs:
    outputs/demo/cloud_rgb.png      — Original RGB
    outputs/demo/cloud_mask.png     — Black = cloud, white = clear
    outputs/demo/cloud_masked.png   — RGB with clouds replaced by red

Options:
    --scl-path PATH   Path to a separate SCL GeoTIFF (optional)
    --output-dir DIR  Output directory for PNGs (default: outputs/demo)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from goldmine_watch.data.cloud_mask import (
    compute_valid_fraction,
    create_cloud_mask,
    load_scl_band,
)


def _stretch_rgb(bands: np.ndarray) -> np.ndarray:
    """Normalize multi-band array to 0-255 uint8 for display."""
    rgb = bands[:3].transpose(1, 2, 0).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


def main() -> int:
    """Run the Feature 2 cloud masking demo."""
    parser = argparse.ArgumentParser(description="Visualize cloud masking on a Sentinel-2 scene")
    parser.add_argument("image_path", type=Path, help="Path to the Sentinel-2 GeoTIFF")
    parser.add_argument("--scl-path", type=Path, default=None, help="Path to separate SCL GeoTIFF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo"),
        help="Directory to save output PNGs",
    )
    parser.add_argument(
        "--invalid-classes",
        type=int,
        nargs="+",
        default=[0, 3, 8, 9],
        help="SCL classes to mask as invalid",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"Error: Image not found: {args.image_path}", file=sys.stderr)
        return 1

    # Load SCL band
    try:
        scl = load_scl_band(args.image_path, scl_path=args.scl_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Create cloud mask
    mask = create_cloud_mask(scl, invalid_classes=args.invalid_classes)
    valid_frac = compute_valid_fraction(mask)
    cloudy_frac = 1.0 - valid_frac

    print(f"Image valid fraction: {valid_frac * 100:.1f}%")
    print(f"Cloudy pixels: {cloudy_frac * 100:.1f}%")

    # Load first 3 bands for RGB visualization
    import rasterio

    with rasterio.open(args.image_path) as src:
        rgb_bands = src.read([1, 2, 3])

    rgb = _stretch_rgb(rgb_bands)

    # Save original RGB
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output_dir / "cloud_rgb.png")

    # Save cloud mask (white = clear, black = cloud)
    mask_img = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_img, mode="L").save(args.output_dir / "cloud_mask.png")

    # Save masked RGB (clouds replaced by red)
    masked_rgb = rgb.copy()
    masked_rgb[mask == 0] = [255, 0, 0]
    Image.fromarray(masked_rgb).save(args.output_dir / "cloud_masked.png")

    print(f"Saved outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
