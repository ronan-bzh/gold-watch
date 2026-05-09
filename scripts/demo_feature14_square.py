#!/usr/bin/env python3
"""Demo script for Feature 14: Square Post-Processing.

Usage:
    python scripts/demo_feature14_square.py \
      --predictions outputs/phase2/ \
      --grid-size 128 \
      --threshold 0.2
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import rasterio
import yaml

from goldmine_watch.data.square_postprocess import square_postprocess


def _load_postprocess_defaults(config_path: str = "configs/mvp.yaml") -> dict:
    """Load postprocess defaults from the centralized config file."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("postprocess", {}) if raw else {}


def main() -> None:
    """Parse arguments and run the Feature 14 square post-processing demo."""
    defaults = _load_postprocess_defaults()

    parser = argparse.ArgumentParser(description="Demo: Square Post-Processing")
    parser.add_argument(
        "--predictions",
        default="outputs/phase2",
        help="Directory containing prediction GeoTIFFs",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=defaults.get("grid_size_m", 128.0),
        help="Grid cell size in meters",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=defaults.get("threshold", 0.2),
        help="Probability threshold for detections",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=defaults.get("min_confidence", 0.3),
        help="Minimum confidence for filtering",
    )
    parser.add_argument(
        "--output",
        default="outputs/detections_square.geojson",
        help="Output GeoJSON path",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions)
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    raster_paths = sorted(predictions_dir.glob("*.tif"))
    if not raster_paths:
        raise ValueError(f"No .tif files found in {predictions_dir}")

    print("Square Post-Processing")
    print("======================")
    print(f"Merging {len(raster_paths)} prediction raster(s)...")

    with rasterio.open(raster_paths[0]) as src:
        h, w = src.height, src.width
        crs = src.crs

    print(f"Mosaic: {w}x{h} px | {crs}")
    print(f"\nOverlaying {args.grid_size}m grid...")

    start = time.time()
    result = square_postprocess(
        probability_raster_paths=raster_paths,
        grid_size_m=args.grid_size,
        threshold=args.threshold,
        min_confidence=args.min_confidence,
        output_path=args.output,
    )
    elapsed = time.time() - start

    gdf = gpd.read_file(result)

    print(f"\nFiltering by threshold (>= {args.threshold}):")
    print(f"  Detections: {len(gdf)}")
    if len(gdf) > 0:
        print(f"  Mean confidence: {gdf['confidence'].mean():.2f}")
        print(f"  Max confidence: {gdf['confidence'].max():.2f}")

    print(f"\nSaved {len(gdf)} square detection(s) to {result}")
    print(f"Processing time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
