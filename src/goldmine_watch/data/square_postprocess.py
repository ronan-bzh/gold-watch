"""Square post-processing for Feature 14.

Convert probability heatmaps into square bounding box detections on a fixed grid.
"""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.merge import merge
from shapely.geometry import Polygon


def create_square_grid(
    bounds: tuple[float, float, float, float],
    grid_size_m: float,
    crs: str = "EPSG:2972",
) -> gpd.GeoDataFrame:
    """Create a regular grid of square cells.

    Args:
        bounds: (min_x, min_y, max_x, max_y) bounding box.
        grid_size_m: Cell size in meters.
        crs: Coordinate reference system string.

    Returns:
        GeoDataFrame with square cell geometries.
    """
    min_x, min_y, max_x, max_y = bounds

    x_coords = np.arange(min_x, max_x + grid_size_m, grid_size_m)
    y_coords = np.arange(min_y, max_y + grid_size_m, grid_size_m)

    cells: list[Polygon] = []
    for x in x_coords:
        for y in y_coords:
            cell = Polygon(
                [
                    (x, y),
                    (x + grid_size_m, y),
                    (x + grid_size_m, y + grid_size_m),
                    (x, y + grid_size_m),
                ]
            )
            cells.append(cell)

    gdf = gpd.GeoDataFrame(geometry=cells, crs=crs)
    return gdf


def _cell_confidence(
    src: rasterio.DatasetReader,
    cell_geom: Polygon,
) -> float:
    """Compute mean probability within a grid cell from an open raster.

    Args:
        src: Open rasterio dataset reader.
        cell_geom: Shapely polygon representing the grid cell.

    Returns:
        Mean probability value within the cell, or 0.0 if no valid data.
    """
    try:
        masked, _ = rasterio_mask(
            src, [cell_geom], crop=True, all_touched=True, filled=False
        )
        if np.ma.count(masked) == 0:
            return 0.0
        return float(np.ma.mean(masked))
    except ValueError:
        return 0.0


def _merge_probability_rasters(
    raster_paths: list[Path], output_path: Path
) -> Path:
    """Merge multiple probability rasters into a single mosaic.

    Args:
        raster_paths: List of probability raster paths.
        output_path: Path to save the merged mosaic.

    Returns:
        Path to the merged mosaic raster.
    """
    if len(raster_paths) == 1:
        return raster_paths[0]

    with contextlib.ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(p)) for p in raster_paths]
        mosaic, transform = merge(datasets)

        crs = datasets[0].crs
        dtype = datasets[0].dtypes[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mosaic[0], 1)

    return output_path


def square_postprocess(
    probability_raster_paths: list[Path],
    grid_size_m: float = 128.0,
    threshold: float = 0.2,
    min_confidence: float = 0.3,
    output_path: str = "outputs/detections_square.geojson",
) -> Path:
    """Convert probability rasters into square bounding boxes.

    Steps:
        1. Merge all probability rasters into a mosaic.
        2. Overlay a fixed-size grid (e.g., 128m cells).
        3. For each cell: compute mean probability.
        4. If mean >= threshold: emit square GeoJSON feature.
        5. Filter by minimum confidence.

    Args:
        probability_raster_paths: List of paths to probability GeoTIFFs.
        grid_size_m: Grid cell size in meters.
        threshold: Mean probability threshold for emitting a detection.
        min_confidence: Minimum confidence value for filtering detections.
        output_path: Path to save the output GeoJSON.

    Returns:
        Path to the saved GeoJSON file.
    """
    if not probability_raster_paths:
        raise ValueError("At least one probability raster path is required.")
    if grid_size_m <= 0:
        raise ValueError("grid_size_m must be positive")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be in [0, 1]")
    if not (0 <= min_confidence <= 1):
        raise ValueError("min_confidence must be in [0, 1]")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge rasters
    if len(probability_raster_paths) == 1:
        merged_path = probability_raster_paths[0]
    else:
        merged_path = output_path_obj.parent / f"_mosaic_{uuid.uuid4().hex}.tif"
        merged_path = _merge_probability_rasters(probability_raster_paths, merged_path)

    # Step 2-4: Open raster once, compute confidences, filter
    with rasterio.open(merged_path) as src:
        bounds = src.bounds
        crs = src.crs

        grid = create_square_grid(
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            grid_size_m=grid_size_m,
            crs=str(crs) if crs is not None else "EPSG:2972",
        )

        confidences: list[float] = [
            _cell_confidence(src, geom) for geom in grid.geometry
        ]

    grid["confidence"] = confidences
    # Apply both thresholds; use the stricter of the two
    effective_threshold = max(threshold, min_confidence)
    detections = grid[grid["confidence"] >= effective_threshold].copy()

    if len(detections) == 0:
        gdf = gpd.GeoDataFrame(
            {
                "detection_id": pd.Series([], dtype=int),
                "confidence": pd.Series([], dtype=float),
                "area_m2": pd.Series([], dtype=float),
            },
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs=grid.crs,
        )
    else:
        detections["detection_id"] = range(1, len(detections) + 1)
        detections["area_m2"] = detections.geometry.area
        gdf = detections[["detection_id", "confidence", "area_m2", "geometry"]]

    gdf.to_file(output_path_obj, driver="GeoJSON")

    # Clean up temporary mosaic
    if len(probability_raster_paths) > 1 and merged_path.exists():
        merged_path.unlink()

    return output_path_obj
