"""Demo script for Feature 1: Real Data Ingestion & Validation.

Usage:
    python scripts/demo_feature1_validate.py \
        data/raw/sentinel2_scene.tif \
        data/french_guiana_mines_2023.geojson
"""

import argparse
import sys
from pathlib import Path

from goldmine_watch.data.validate import (
    check_spatial_overlap,
    validate_image,
    validate_labels,
)


def main() -> int:
    """Validate an image and labels, printing a summary."""
    parser = argparse.ArgumentParser(
        description="Validate a Sentinel-2 scene and mining surface labels."
    )
    parser.add_argument("image_path", type=Path, help="Path to the Sentinel-2 GeoTIFF")
    parser.add_argument(
        "labels_path",
        type=Path,
        help="Path to the mining labels (GeoPackage, GeoJSON, Shapefile)",
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=6,
        help="Expected band count (default: 6 for raw Sentinel-2 scenes)",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"Error: Image not found: {args.image_path}", file=sys.stderr)
        return 1

    if not args.labels_path.exists():
        print(f"Error: Labels not found: {args.labels_path}", file=sys.stderr)
        return 1

    # Validate image
    try:
        img_meta = validate_image(args.image_path, expected_band_count=args.bands)
    except ValueError as exc:
        print(f"Error: Image validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"✅ Image: {img_meta['width']}x{img_meta['height']} pixels, "
        f"{img_meta['band_count']} bands, {img_meta['crs']}"
    )

    # Validate labels
    try:
        labels_gdf = validate_labels(args.labels_path, expected_crs=img_meta["crs"])
    except ValueError as exc:
        print(f"Error: Labels validation failed: {exc}", file=sys.stderr)
        return 1

    source_crs = labels_gdf.crs.to_string() if labels_gdf.crs else "unknown"
    reprojected = source_crs != img_meta["crs"]
    reproj_note = f" (reprojected from {source_crs})" if reprojected else ""

    print(f"✅ Labels: {len(labels_gdf)} polygons, {img_meta['crs']}{reproj_note}")

    # Check spatial overlap
    overlap_stats = check_spatial_overlap(img_meta["bounds"], labels_gdf)

    total = overlap_stats["total_labels"]
    inside = overlap_stats["overlapping_labels"]
    outside = overlap_stats["outside_labels"]
    pct = overlap_stats["overlap_fraction"] * 100

    if overlap_stats["has_overlap"]:
        print(
            f"✅ Overlap: {inside}/{total} labels ({pct:.0f}%) intersect image bounds"
        )
        if outside > 0:
            print(
                f"⚠️  {outside} label{'s' if outside > 1 else ''} fall"
                f" outside image — review coordinates"
            )
    else:
        print(f"⚠️  Overlap: no labels intersect image bounds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
