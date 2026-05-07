"""Post-processing: convert probability raster to vector polygons.

Milestone 10: Can I turn predictions into polygons?
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape


def postprocess(
    probability_raster_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    min_area_pixels: int = 10,
) -> Path:
    """Threshold a probability raster and extract polygons.

    Args:
        probability_raster_path: Path to the probability GeoTIFF.
        output_path: Path to save the output GeoPackage.
        threshold: Probability threshold for binarization.
        min_area_pixels: Minimum connected component area in pixels.

    Returns:
        Path to the saved GeoPackage.
    """
    probability_raster_path = Path(probability_raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(probability_raster_path) as src:
        probs = src.read(1)
        transform = src.transform
        crs = src.crs

    binary = (probs >= threshold).astype(np.uint8)

    shapes_gen = features.shapes(binary, mask=binary == 1, transform=transform)
    polygons = []
    for geom, val in shapes_gen:
        if val == 1:
            polygons.append(shape(geom))

    if not polygons:
        print("Warning: no polygons found above threshold.")
        gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
    else:
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        # Filter by area
        gdf["area_px"] = gdf.geometry.area / (transform.a**2)
        gdf = gdf[gdf["area_px"] >= min_area_pixels]
        gdf = gdf.drop(columns=["area_px"])

    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved {len(gdf)} polygons to {output_path}")
    return output_path
