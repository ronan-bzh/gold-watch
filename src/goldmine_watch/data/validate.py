"""Data validation: verify images, labels, and spatial alignment."""

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
from shapely.geometry import box

logger = logging.getLogger(__name__)

EXPECTED_BAND_COUNT = 9


def validate_image(
    image_path: Path, expected_band_count: int | None = None
) -> dict[str, Any]:
    """Check a GeoTIFF has expected bands, CRS, and resolution.

    Args:
        image_path: Path to the GeoTIFF image.
        expected_band_count: Expected number of bands. Defaults to
            :data:`EXPECTED_BAND_COUNT` (9) when *None*.

    Returns:
        Dict with keys: crs, band_count, width, height, resolution, bounds.

    Raises:
        ValueError: If band count doesn't match, CRS is undefined, or
            resolution is inconsistent/invalid.
    """
    if expected_band_count is None:
        expected_band_count = EXPECTED_BAND_COUNT

    with rasterio.open(image_path) as src:
        band_count = src.count
        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform
        bounds = src.bounds

    if band_count != expected_band_count:
        raise ValueError(
            f"Expected {expected_band_count} bands, got {band_count} in {image_path}"
        )

    if crs is None:
        raise ValueError(f"CRS is undefined for {image_path}")

    # Extract resolution from the affine transform
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    if res_x <= 0 or res_y <= 0:
        raise ValueError(f"Invalid resolution ({res_x}, {res_y}) in {image_path}")

    if abs(res_x - res_y) > 1e-6:
        raise ValueError(
            f"Non-square pixels: x_res={res_x}, y_res={res_y} in {image_path}"
        )

    return {
        "crs": crs.to_string(),
        "band_count": band_count,
        "width": width,
        "height": height,
        "resolution": res_x,
        "bounds": bounds,
    }


def validate_labels(labels_path: Path, expected_crs: str) -> gpd.GeoDataFrame:
    """Load labels and verify CRS matches expected.

    Reprojects if necessary. Raises ValueError if empty or invalid.

    Args:
        labels_path: Path to the vector file (GeoPackage, GeoJSON, Shapefile, etc.).
        expected_crs: Expected CRS string (e.g. "EPSG:2972").

    Returns:
        Cleaned GeoDataFrame reprojected to expected_crs.

    Raises:
        ValueError: If the file contains no valid geometries.
    """
    gdf = gpd.read_file(labels_path)

    # Drop None, empty, and invalid geometries
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.is_valid]

    if len(gdf) == 0:
        raise ValueError(f"No valid geometries found in {labels_path}")

    # Reproject if necessary
    if gdf.crs is None:
        raise ValueError(f"CRS is undefined for {labels_path}")

    if not gdf.crs.to_string() == expected_crs:
        gdf = gdf.to_crs(expected_crs)

    return gdf


def check_spatial_overlap(
    image_bounds: Any, labels_gdf: gpd.GeoDataFrame
) -> dict[str, Any]:
    """Check if labels intersect with image bounds and return detailed stats.

    Args:
        image_bounds: Raster bounds (left, bottom, right, top) or rasterio
            BoundingBox.
        labels_gdf: GeoDataFrame with polygon geometries.

    Returns:
        Dict with keys:
            - has_overlap (bool): True if at least one label intersects.
            - total_labels (int): Total number of labels.
            - overlapping_labels (int): Number of labels intersecting the image.
            - overlap_fraction (float): Fraction of total label area inside
              the image bounds (0.0–1.0).
            - outside_labels (int): Number of labels completely outside.
    """
    image_box = box(
        image_bounds.left, image_bounds.bottom, image_bounds.right, image_bounds.top
    )

    total_label_area = 0.0
    overlapping_area = 0.0
    overlap_count = 0
    outside_count = 0

    for geom in labels_gdf.geometry:
        if geom.is_empty:
            continue
        label_area = geom.area
        total_label_area += label_area

        if image_box.intersects(geom):
            overlap_count += 1
            intersection = image_box.intersection(geom)
            overlapping_area += intersection.area
        else:
            outside_count += 1

    overlap_fraction = (
        overlapping_area / total_label_area if total_label_area > 0 else 0.0
    )

    if overlap_fraction < 0.5:
        logger.warning(
            "Only %.1f%% of label area overlaps with the image bounds",
            overlap_fraction * 100,
        )

    return {
        "has_overlap": overlap_count > 0,
        "total_labels": len(labels_gdf),
        "overlapping_labels": overlap_count,
        "overlap_fraction": overlap_fraction,
        "outside_labels": outside_count,
    }
