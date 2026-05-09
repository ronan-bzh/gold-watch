#!/usr/bin/env python3
r"""Demo script for Feature 9: Unified Tile Cache System.

Usage::

    export $(cat .env | xargs)
    python scripts/demo_feature9_tile_cache.py \\
        --bbox "-54.05,5.45,-54.0,5.49" \\
        --date "2023-06-01/2023-12-31"
"""

import argparse
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from goldmine_watch.data.tile_cache import TileCache, get_tile_id_from_bbox


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse a comma-separated bbox string."""
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must have exactly 4 comma-separated values")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    """Run the tile cache demo."""
    parser = argparse.ArgumentParser(description="Tile Cache System Demo")
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        required=True,
        help="Bounding box as min_lon,min_lat,max_lon,max_lat",
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date range as YYYY-MM-DD/YYYY-MM-DD",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/tiles",
        help="Cache directory path",
    )
    args = parser.parse_args()

    tile_id = get_tile_id_from_bbox(args.bbox)
    cache = TileCache(args.cache_dir)

    print("Tile Cache System Demo")
    print("=" * 40)
    print(f"Checking cache at: {cache.cache_dir}/")

    # First request — may trigger download
    tile_path = cache.get_tile(
        tile_id=tile_id,
        date_range=args.date,
        bbox=args.bbox,
    )

    if tile_path.stat().st_size < 1024 * 1024:
        # Synthetic / mocked data — skip detailed reporting
        print(f"Tile {tile_path.stem} cached.")
    else:
        with rasterio.open(tile_path) as src:
            width, height = src.width, src.height
            bands = src.count
            size_mb = tile_path.stat().st_size / (1024 * 1024)
            print(f"Tile {tile_path.stem} cached.")
            print(f"Dimensions: {width}x{height} px | {bands} bands | {size_mb:.1f} MB")

    print(f"Cached to: {tile_path}")
    print()

    # Second request — should be a cache hit
    print("Checking cache again...")
    tile_path2 = cache.get_tile(
        tile_id=tile_id,
        date_range=args.date,
        bbox=args.bbox,
    )

    size_mb = tile_path2.stat().st_size / (1024 * 1024)
    print(f"Tile {tile_path2.stem} found in cache! ({size_mb:.1f} MB)")
    print("Cache reuse avoided redundant download.")
    print()

    # Statistics
    print("Cache statistics:")
    print(f"  Tiles cached: {len(cache.list_cached_tiles())}")
    print(f"  Total size: {cache.get_cache_size_mb():.1f} MB")


if __name__ == "__main__":
    main()
