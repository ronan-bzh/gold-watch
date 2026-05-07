#!/usr/bin/env python3
"""Demo script for Feature 0: Copernicus Data Space Download.

Usage:
    export COPERNICUS_CLIENT_ID="your-client-id"
    export COPERNICUS_CLIENT_SECRET="your-client-secret"
    python scripts/demo_feature0_download.py \
      --bbox "-54.1,5.3,-53.9,5.5" \
      --date "2023-01-01/2023-01-31" \
      --output data/raw/sentinel2_scene.tif
"""

import argparse
import os
from pathlib import Path

import rasterio

from goldmine_watch.data.copernicus import download_scene, get_access_token, search_scenes


def main() -> None:
    """Run the Feature 0 download demo."""
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 scenes from Copernicus Data Space"
    )
    parser.add_argument(
        "--bbox", required=True, help="Bounding box as min_x,min_y,max_x,max_y"
    )
    parser.add_argument(
        "--date", required=True, help="Date range as YYYY-MM-DD/YYYY-MM-DD"
    )
    parser.add_argument("--output", required=True, help="Output GeoTIFF path")
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum cloud cover percentage",
    )
    parser.add_argument(
        "--bands",
        default="B02,B03,B04,B08,B11,B12",
        help="Comma-separated band list",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="Target resolution in meters",
    )
    args = parser.parse_args()

    client_id = os.environ.get("COPERNICUS_CLIENT_ID")
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit(
            "Error: Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET environment variables"
        )

    bbox = tuple(float(x) for x in args.bbox.split(","))
    bands = args.bands.split(",")
    output_path = Path(args.output)

    print("Authenticating with Copernicus Data Space...")
    token = get_access_token(client_id, client_secret)

    print(
        f"Search: bbox={list(bbox)} date={args.date} max_cloud={args.max_cloud}%"
    )
    scenes = search_scenes(
        bbox, args.date, token, max_cloud_cover=args.max_cloud
    )

    print(f"Found {len(scenes)} scenes:")
    for i, scene in enumerate(scenes, 1):
        cloud = scene.get("properties", {}).get("eo:cloud_cover", "unknown")
        print(f"  {i}. {scene['id']}  cloud: {cloud}%")

    item = scenes[0]
    cloud_cover = item.get("properties", {}).get("eo:cloud_cover", "unknown")
    print(f"Downloading scene 1 ({len(bands)} bands, {args.resolution}m)...")
    saved_path = download_scene(
        item,
        token,
        output_path,
        bands=bands,
        resolution=args.resolution,
        bbox=bbox,
    )

    with rasterio.open(saved_path) as src:
        print(f"Saved to {saved_path}")
        print(
            f"{src.width}x{src.height} pixels | {src.count} bands | "
            f"{src.crs} | {cloud_cover}% cloud cover"
        )


if __name__ == "__main__":
    main()
