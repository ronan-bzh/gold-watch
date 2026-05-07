"""Data ingestion: load vector labels and rasterize them to binary masks."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features


def load_labels(labels_path: str | Path, target_crs: str | None = None) -> gpd.GeoDataFrame:
    """Load mining surface labels from a vector file.

    Drops invalid and empty geometries. Optionally reprojects to a target CRS.
    Raises ValueError if no valid geometries remain after cleaning.

    Args:
        labels_path: Path to the vector file (GeoPackage, Shapefile, GeoJSON, etc.).
        target_crs: Optional CRS to reproject to (e.g. "EPSG:2972").

    Returns:
        Cleaned GeoDataFrame with valid geometries.

    Raises:
        ValueError: If the file contains no valid geometries after cleaning.
    """
    gdf = gpd.read_file(labels_path)

    # Drop empty geometries
    gdf = gdf[~gdf.geometry.is_empty]

    # Drop invalid geometries
    gdf = gdf[gdf.geometry.is_valid]

    if len(gdf) == 0:
        raise ValueError(f"No valid geometries found in {labels_path}")

    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    return gdf


def burn_mask(
    labels_gdf: gpd.GeoDataFrame,
    reference_raster_path: str | Path,
) -> np.ndarray:
    """Rasterize polygon labels into a binary mask matching a reference raster.

    Args:
        labels_gdf: GeoDataFrame with polygon geometries in the same CRS as the raster.
        reference_raster_path: Path to the reference raster (GeoTIFF) whose shape,
            transform, and CRS define the output mask.

    Returns:
        2D numpy array of shape (height, width) with 1 where labels exist and 0 elsewhere.
    """
    with rasterio.open(reference_raster_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    # Ensure labels are in the same CRS as the raster
    if labels_gdf.crs is not None and crs is not None and not labels_gdf.crs.equals(crs):
        labels_gdf = labels_gdf.to_crs(crs)

    shapes = ((geom, 1) for geom in labels_gdf.geometry if not geom.is_empty)
    mask = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    return np.asarray(mask, dtype=np.uint8)
