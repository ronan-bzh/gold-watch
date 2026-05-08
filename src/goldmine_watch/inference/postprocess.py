"""Post-processing: convert probability raster to vector polygons.

Milestone 10: Can I turn predictions into polygons?
Feature 7: Post-processing, polygonization, and export.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import shape


def postprocess(
    probability_raster_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    min_area_pixels: int = 10,
    min_area_m2: float | None = None,
) -> Path:
    """Threshold a probability raster and extract polygons.

    Args:
        probability_raster_path: Path to the probability GeoTIFF.
        output_path: Path to save the output GeoPackage.
        threshold: Probability threshold for binarization.
        min_area_pixels: Minimum connected component area in pixels.
        min_area_m2: Optional minimum real-world area in square meters.

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
        gdf = gpd.GeoDataFrame(
            {
                "detection_id": pd.Series([], dtype=int),
                "confidence": pd.Series([], dtype=float),
                "area_m2": pd.Series([], dtype=float),
                "area_ha": pd.Series([], dtype=float),
            },
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs=crs,
        )
    else:
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        gdf["area_px"] = gdf.geometry.area / (transform.a**2)
        gdf = gdf[gdf["area_px"] >= min_area_pixels]
        gdf = gdf.drop(columns=["area_px"])

        if min_area_m2 is not None:
            gdf = gdf[gdf.geometry.area >= min_area_m2]

        # Compute area and confidence
        gdf["detection_id"] = range(1, len(gdf) + 1)
        gdf["area_m2"] = gdf.geometry.area
        gdf["area_ha"] = gdf["area_m2"] / 10_000.0

        confidences: list[float] = []
        with rasterio.open(probability_raster_path) as src:
            for geom in gdf.geometry:
                try:
                    masked, _ = rasterio_mask(
                        src, [geom], crop=True, all_touched=True, filled=False
                    )
                    if np.ma.count(masked) == 0:
                        confidences.append(0.0)
                    else:
                        confidences.append(float(np.ma.mean(masked)))
                except ValueError:
                    confidences.append(0.0)
        gdf["confidence"] = confidences

        # Reorder columns
        gdf = gdf[["detection_id", "confidence", "area_m2", "area_ha", "geometry"]]

    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved {len(gdf)} polygons to {output_path}")
    return output_path
