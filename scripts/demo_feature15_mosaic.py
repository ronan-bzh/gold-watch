#!/usr/bin/env python3
"""Demo script for Feature 15: Mosaic Builder.

Usage:
    python scripts/demo_feature15_mosaic.py \
      --input outputs/phase2/ \
      --output outputs/phase2/mosaic.tif \
      --method mean
"""

import argparse
import time
from pathlib import Path

import rasterio
import yaml

from goldmine_watch.data.mosaic import build_mosaic, validate_mosaic


def _load_mosaic_defaults(config_path: str = "configs/mvp.yaml") -> dict:
    """Load mosaic defaults from the centralized config file."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("mosaic", {}) if raw else {}


def main() -> None:
    """Parse arguments and run the Feature 15 mosaic builder demo."""
    # Parse --config first so defaults can be loaded from the specified file.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="configs/mvp.yaml")
    pre_args, remaining = pre_parser.parse_known_args()
    defaults = _load_mosaic_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="Demo: Mosaic Builder")
    parser.add_argument(
        "--input",
        default="outputs/phase2",
        help="Directory containing prediction GeoTIFFs",
    )
    parser.add_argument(
        "--output",
        default=defaults.get("output_path", "outputs/phase2/mosaic.tif"),
        help="Output mosaic path",
    )
    parser.add_argument(
        "--method",
        default=defaults.get("method", "mean"),
        choices=["mean", "max"],
        help="Merge method for overlapping tiles",
    )
    parser.add_argument(
        "--config",
        default=pre_args.config,
        help="Path to configuration file",
    )
    args = parser.parse_args(remaining)

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    raster_paths = sorted(input_dir.glob("*.tif"))
    if not raster_paths:
        raise ValueError(f"No .tif files found in {input_dir}")

    print("Mosaic Builder")
    print("==============")
    print(f"Loading {len(raster_paths)} prediction raster(s)...")
    for p in raster_paths:
        with rasterio.open(p) as src:
            print(f"  {p.stem}: {src.width}x{src.height} px")

    print(f"\nMerging with method: {args.method}")
    start = time.time()
    result = build_mosaic(
        raster_paths=raster_paths,
        output_path=args.output,
        method=args.method,
    )
    elapsed = time.time() - start

    with rasterio.open(result) as src:
        print(f"Output: {src.width}x{src.height} px | {src.crs}")
        file_size_mb = result.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")

    print("\nValidation:")
    report = validate_mosaic(result)
    if report["has_gaps"]:
        print(f"  WARNING: {report['gap_count']} gap(s) detected")
    else:
        print("  No gaps detected")
    print(f"  Value range: [{report['min_value']:.3f}, {report['max_value']:.3f}]")
    if report["out_of_range_count"] > 0:
        print(f"  WARNING: {report['out_of_range_count']} out-of-range pixel(s)")

    print(f"\nSaved to {result}")
    print(f"Processing time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
