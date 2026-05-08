#!/usr/bin/env python3
"""Demo script for Feature 6: Temporal Compositing.

Usage:
    python scripts/demo_feature6_composite.py \
      --bbox "-54.1,5.3,-53.9,5.5" \
      --start 2023-01-01 \
      --end 2023-03-31 \
      --out data/raw/composite.tif
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from goldmine_watch.data.stac import download_composite, download_one_scene


def _read_rgb(image_path: Path) -> np.ndarray:
    """Read and normalize an RGB image for display.

    Tries to use bands 3, 2, 1 (B04, B03, B02) for true-color if available,
    otherwise falls back to the first three bands.
    """
    with rasterio.open(image_path) as src:
        count = src.count
        if count >= 3:
            # Prefer true-color ordering if we have at least 3 bands
            rgb = src.read([3, 2, 1]).astype(np.float32)
        elif count == 2:
            bands = src.read([1, 2]).astype(np.float32)
            rgb = np.concatenate([bands, bands[1:2]], axis=0)
        elif count == 1:
            band = src.read(1).astype(np.float32)
            rgb = np.stack([band] * 3, axis=0)
        else:
            raise ValueError(f"Unexpected band count: {count}")

    rgb = rgb.transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    return np.asarray(rgb, dtype=np.float32)


def main() -> None:
    """Run the Feature 6 composite demo."""
    parser = argparse.ArgumentParser(
        description="Build a cloud-free temporal composite",
        epilog=(
            "Example: python %(prog)s --bbox=\"-54.1,5.3,-53.9,5.5\" "
            "--start 2023-01-01 --end 2023-03-31 --out composite.tif"
        ),
    )
    parser.add_argument(
        "--bbox",
        required=True,
        help='Bounding box as "min_x,min_y,max_x,max_y" (quotes required if negative)',
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", required=True, help="Output composite GeoTIFF path")
    parser.add_argument(
        "--max-cloud", type=float, default=20.0, help="Maximum cloud cover per scene"
    )
    parser.add_argument(
        "--aggregator",
        default="median",
        choices=["median", "mean"],
        help="Compositing aggregator",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/demo",
        help="Directory for comparison figure",
    )
    args = parser.parse_args()

    raw_bbox = [float(x) for x in args.bbox.split(",")]
    if len(raw_bbox) != 4:
        raise ValueError("bbox must have exactly 4 values: min_x,min_y,max_x,max_y")
    bbox: tuple[float, float, float, float] = tuple(raw_bbox)  # type: ignore[assignment]
    output_path = Path(args.out)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build composite
    composite_path = download_composite(
        bbox=bbox,
        start_date=args.start,
        end_date=args.end,
        output_path=output_path,
        max_cloud_cover=args.max_cloud,
        aggregator=args.aggregator,
    )

    # Download a single scene for comparison
    print("Downloading single scene for comparison...")
    single_scene_path = output_dir / "single_scene.tif"
    try:
        download_one_scene(
            bbox=bbox,
            date=f"{args.start}/{args.end}",
            output_path=single_scene_path,
            max_cloud_cover=args.max_cloud,
        )
    except RuntimeError as exc:
        print(f"Could not download single scene for comparison: {exc}")
        return

    # Create comparison figure
    print("Creating comparison figure...")
    composite_rgb = _read_rgb(composite_path)
    single_rgb = _read_rgb(single_scene_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(single_rgb)
    axes[0].set_title("Single Scene (with clouds)")
    axes[0].axis("off")

    axes[1].imshow(composite_rgb)
    axes[1].set_title(f"{args.aggregator.capitalize()} Composite (cloud-free)")
    axes[1].axis("off")

    fig.tight_layout()
    comparison_path = output_dir / "composite_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
