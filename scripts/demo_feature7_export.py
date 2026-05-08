#!/usr/bin/env python3
"""Demo script for Feature 7: Post-Processing, Polygonization & QGIS Export.

Usage:
    python scripts/demo_feature7_export.py \
        data/raw/sentinel2_scene.tif \
        outputs/real_prediction.tif \
        models/best_model.pth \
        --threshold 0.5 \
        --min-area-m2 500
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

from goldmine_watch.export.csv import export_polygon_metrics
from goldmine_watch.export.qgis import create_qgis_project
from goldmine_watch.inference.postprocess import postprocess


def main() -> None:
    """Parse arguments and run post-processing export demo."""
    parser = argparse.ArgumentParser(
        description="Post-process predictions and export polygons + QGIS project"
    )
    parser.add_argument("image", help="Input satellite image GeoTIFF")
    parser.add_argument("prediction", help="Predicted probability GeoTIFF")
    parser.add_argument("model", help="Model checkpoint path (unused, for consistency)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    parser.add_argument(
        "--min-area-m2", type=float, default=None, help="Minimum polygon area in m²"
    )
    parser.add_argument(
        "--min-area-pixels", type=int, default=10, help="Minimum polygon area in pixels"
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    image_path = Path(args.image)
    prediction_path = Path(args.prediction)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prediction_path.exists():
        print(f"Error: prediction raster not found: {prediction_path}", file=sys.stderr)
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: model checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Post-process
    print(f"Thresholding at {args.threshold}...")
    polygons_path = output_dir / "real_polygons.gpkg"
    postprocess(
        prediction_path,
        polygons_path,
        threshold=args.threshold,
        min_area_pixels=args.min_area_pixels,
        min_area_m2=args.min_area_m2,
    )

    # Export CSV
    csv_path = output_dir / "real_polygons.csv"
    export_polygon_metrics(polygons_path, csv_path)
    print(f"Saved CSV: {csv_path}")

    # Create QGIS project
    qgis_path = output_dir / "detection_project.qgz"
    create_qgis_project(image_path, prediction_path, polygons_path, qgis_path)
    print(f"Created QGIS project: {qgis_path}")

    # Summary
    gdf = gpd.read_file(polygons_path)
    total_area_ha = gdf["area_ha"].sum() if len(gdf) > 0 else 0.0
    print(f"Total detected area: {total_area_ha:.1f} ha")


if __name__ == "__main__":
    main()
