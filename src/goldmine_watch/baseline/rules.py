"""Spectral rule-based baseline for mining detection.

Uses NDVI + BSI thresholds to flag bare soil with low vegetation.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - RED) / (NIR + RED)
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    denom = np.where(denom == 0, 1e-8, denom)
    return (nir - red) / denom


def compute_bsi(red: np.ndarray, nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Compute Bare Soil Index.

    BSI = ((RED + SWIR1) - (NIR)) / ((RED + SWIR1) + (NIR))
    Simplified version for Sentinel-2.
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    denom = red + swir1 + nir
    denom = np.where(denom == 0, 1e-8, denom)
    return (red + swir1 - nir) / denom


def detect_mining_rules(
    image_path: Path,
    ndvi_threshold: float = 0.2,
    bsi_threshold: float = 0.1,
) -> np.ndarray:
    """Return binary mask where NDVI < threshold AND BSI > threshold.

    Expects a 6+ band GeoTIFF with B04 (RED), B08 (NIR), B11 (SWIR1).
    """
    with rasterio.open(image_path) as src:
        # Try to find bands by description; fallback to fixed indices
        band_names = [src.descriptions[i] or "" for i in range(src.count)]
        red_idx = band_names.index("B04") + 1 if "B04" in band_names else 3
        nir_idx = band_names.index("B08") + 1 if "B08" in band_names else 4
        swir1_idx = band_names.index("B11") + 1 if "B11" in band_names else 5

        red = src.read(red_idx)
        nir = src.read(nir_idx)
        swir1 = src.read(swir1_idx)

    ndvi = compute_ndvi(nir, red)
    bsi = compute_bsi(red, nir, swir1)

    mask = (ndvi < ndvi_threshold) & (bsi > bsi_threshold)
    return mask.astype(np.uint8)


def rules_to_polygons(
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str | rasterio.CRS,
) -> gpd.GeoDataFrame:
    """Convert binary rule mask to vector polygons.

    Args:
        mask: 2D binary array.
        transform: Rasterio affine transform.
        crs: Coordinate reference system.

    Returns:
        GeoDataFrame with extracted polygons.
    """
    shapes_gen = features.shapes(mask, mask=mask == 1, transform=transform)
    polygons = []
    for geom, val in shapes_gen:
        if val == 1:
            polygons.append(shape(geom))

    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    return gpd.GeoDataFrame(geometry=polygons, crs=crs)
