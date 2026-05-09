#!/usr/bin/env python3
"""Demo script for Feature 18: QGIS Export — Full Territory.

Usage:
    python scripts/demo_feature18_qgis.py \
      --mosaic outputs/phase2/mosaic.tif \
      --detections outputs/detections_square.geojson \
      --labels data/french_guiana_mines.geojson \
      --output outputs/goldmine_watch_full.qgz
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import rasterio

from goldmine_watch.export.qgis import create_qgis_project_full


def main() -> None:
    """Parse arguments and run the Feature 18 QGIS export demo."""
    parser = argparse.ArgumentParser(
        description="Export full territory mosaic and detections as a QGIS project"
    )
    parser.add_argument(
        "--mosaic",
        required=True,
        help="Path to the mosaic GeoTIFF (probability heatmap)",
    )
    parser.add_argument(
        "--detections",
        required=True,
        help="Path to detections GeoJSON",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to labels GeoJSON",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional path to RGB composite GeoTIFF",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output QGIS project path",
    )
    args = parser.parse_args()

    mosaic_path = Path(args.mosaic)
    detections_path = Path(args.detections)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    # Validate inputs
    if not mosaic_path.exists():
        print(f"Error: mosaic not found: {mosaic_path}", file=sys.stderr)
        sys.exit(1)

    if not detections_path.exists():
        print(f"Error: detections not found: {detections_path}", file=sys.stderr)
        sys.exit(1)

    if not labels_path.exists():
        print(f"Error: labels not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    image_path = Path(args.image) if args.image else None
    if image_path is not None and not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Gather metadata for summary
    with rasterio.open(mosaic_path) as src:
        mosaic_dims = f"{src.width:,} x {src.height:,} px"

    detections_gdf = gpd.read_file(detections_path)
    labels_gdf = gpd.read_file(labels_path)

    # Build project
    result = create_qgis_project_full(
        mosaic_path=mosaic_path,
        detections_path=detections_path,
        labels_path=labels_path,
        output_project_path=output_path,
        image_path=image_path,
    )

    # Summary
    print("QGIS Export — Full Territory")
    print("============================")
    print("Layers:")
    layer_num = 1
    if image_path is not None:
        print(f"  {layer_num}. Satellite Composite (RGB)")
        layer_num += 1
    print(f"  {layer_num}. Mosaic (probability)  — {mosaic_dims}")
    layer_num += 1
    print(
        f"  {layer_num}. Detections (squares)  — {len(detections_gdf):,} features, red outline"
    )
    layer_num += 1
    print(
        f"  {layer_num}. Labels (squares)      — {len(labels_gdf):,} features, green outline"
    )
    print()
    print(f"Saved to {result}")
    print("Open in QGIS to explore.")


if __name__ == "__main__":
    main()
