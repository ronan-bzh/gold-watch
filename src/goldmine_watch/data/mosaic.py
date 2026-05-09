"""Mosaic builder for Feature 15.

Merge multiple per-tile probability rasters into a single seamless mosaic.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_bounds
from rasterio.warp import reproject

NODATA_FILL = -9999.0


def get_mosaic_bounds(raster_paths: list[Path]) -> tuple[float, float, float, float]:
    """Compute combined bounds of all input rasters.

    Args:
        raster_paths: List of paths to GeoTIFF files.

    Returns:
        (min_x, min_y, max_x, max_y) bounding box.

    Raises:
        ValueError: If *raster_paths* is empty.
    """
    if not raster_paths:
        raise ValueError("At least one raster path is required.")

    bounds_list: list[tuple[float, float, float, float]] = []
    for p in raster_paths:
        with rasterio.open(p) as src:
            b = src.bounds
            bounds_list.append((b.left, b.bottom, b.right, b.top))

    min_x = min(b[0] for b in bounds_list)
    min_y = min(b[1] for b in bounds_list)
    max_x = max(b[2] for b in bounds_list)
    max_y = max(b[3] for b in bounds_list)
    return (min_x, min_y, max_x, max_y)


def _create_cog(src_path: Path, dst_path: Path) -> Path:
    """Convert a tiled GeoTIFF to a Cloud Optimized GeoTIFF.

    Builds overviews and copies using the COG driver.
    """
    with rasterio.open(src_path, "r+") as src:
        factors: list[int] = []
        factor = 2
        while src.width // factor > 1 and src.height // factor > 1:
            factors.append(factor)
            factor *= 2
        if factors:
            src.build_overviews(factors, Resampling.nearest)
            src.update_tags(ns="IMAGE_STRUCTURE", TILED="YES")

    try:
        rio_copy(src_path, dst_path, driver="COG")
    except Exception:
        # Fallback: keep the tiled GeoTIFF with overviews (COG-compatible)
        shutil.copy(str(src_path), str(dst_path))

    return dst_path


def _bounds_from_datasets(
    datasets: list[rasterio.DatasetReader],
) -> tuple[float, float, float, float]:
    """Return the union bounds of already-open raster datasets."""
    bounds_list = [
        (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
        for ds in datasets
    ]
    min_x = min(b[0] for b in bounds_list)
    min_y = min(b[1] for b in bounds_list)
    max_x = max(b[2] for b in bounds_list)
    max_y = max(b[3] for b in bounds_list)
    return (min_x, min_y, max_x, max_y)


def build_mosaic(
    raster_paths: list[Path],
    output_path: str = "outputs/phase2/mosaic.tif",
    method: str = "mean",
) -> Path:
    """Merge multiple GeoTIFFs into a single mosaic.

    Overlapping edges are resolved by averaging (``method="mean"``) or by
    taking the per-pixel maximum (``method="max"``).

    The output is written as a Cloud Optimized GeoTIFF (COG).

    Args:
        raster_paths: List of paths to input GeoTIFFs.
        output_path: Path where the mosaic COG will be saved.
        method: Merge strategy — ``"mean"`` or ``"max"``.

    Returns:
        Path to the saved mosaic file.

    Raises:
        ValueError: If *raster_paths* is empty or *method* is unsupported.
    """
    if not raster_paths:
        raise ValueError("At least one raster path is required.")
    if method not in {"mean", "max"}:
        raise ValueError(f"Unsupported merge method: {method}. Use 'mean' or 'max'.")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(p)) for p in raster_paths]
        crs = datasets[0].crs
        dtype = datasets[0].dtypes[0]

        if method == "max":
            mosaic, transform = merge(datasets, method="max", nodata=NODATA_FILL)
            mosaic = mosaic[0]
            nodata_val = NODATA_FILL
        else:
            # Manual mean accumulation using reproject for fine-grained control.
            bounds = _bounds_from_datasets(datasets)
            res_x = abs(datasets[0].transform[0])
            res_y = abs(datasets[0].transform[4])

            width = int(np.ceil((bounds[2] - bounds[0]) / res_x))
            height = int(np.ceil((bounds[3] - bounds[1]) / res_y))
            dst_transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

            sum_arr = np.zeros((height, width), dtype=np.float64)
            count_arr = np.zeros((height, width), dtype=np.uint16)

            for ds in datasets:
                temp = np.empty((height, width), dtype=np.float32)

                reproject(
                    source=rasterio.band(ds, 1),
                    destination=temp,
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    src_nodata=ds.nodata,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    dst_nodata=NODATA_FILL,
                    resampling=Resampling.nearest,
                )

                valid = temp != NODATA_FILL
                sum_arr[valid] += temp[valid]
                count_arr[valid] += 1

            with np.errstate(divide="ignore", invalid="ignore"):
                mosaic = np.where(count_arr > 0, sum_arr / count_arr, NODATA_FILL).astype(dtype)
            transform = dst_transform
            nodata_val = NODATA_FILL

        # Write intermediate tiled GeoTIFF
        tmp_path = output_path_obj.with_suffix(".tmp.tif")
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=mosaic.shape[0],
            width=mosaic.shape[1],
            count=1,
            dtype=mosaic.dtype,
            crs=crs,
            transform=transform,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="lzw",
            nodata=nodata_val,
        ) as dst:
            dst.write(mosaic, 1)

        # Convert to COG
        try:
            _create_cog(tmp_path, output_path_obj)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return output_path_obj


def validate_mosaic(mosaic_path: Path) -> dict:
    """Check mosaic for gaps, artifacts, or invalid values.

    Reads the mosaic in block-sized windows to keep memory usage low.

    Args:
        mosaic_path: Path to the mosaic GeoTIFF.

    Returns:
        Dictionary with validation results.
    """
    with rasterio.open(mosaic_path) as src:
        nodata = src.nodata

        min_val = float("inf")
        max_val = float("-inf")
        gap_count = 0
        out_of_range = 0
        total_valid = 0

        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=True)

            if np.ma.is_masked(data):
                block_gaps = int(np.ma.count_masked(data))
                valid = data.compressed()
            elif nodata is not None:
                block_gaps = int(np.sum(data == nodata))
                valid = data[data != nodata]
            else:
                block_gaps = 0
                valid = data.flatten()

            gap_count += block_gaps
            total_valid += valid.size

            if valid.size > 0:
                block_min = float(valid.min())
                block_max = float(valid.max())
                if block_min < min_val:
                    min_val = block_min
                if block_max > max_val:
                    max_val = block_max
                out_of_range += int(np.sum((valid < 0) | (valid > 1)))

    if min_val == float("inf"):
        min_val = float("nan")
    if max_val == float("-inf"):
        max_val = float("nan")

    return {
        "has_gaps": gap_count > 0,
        "gap_count": gap_count,
        "min_value": min_val,
        "max_value": max_val,
        "out_of_range_count": out_of_range,
        "valid": gap_count == 0 and out_of_range == 0,
    }
