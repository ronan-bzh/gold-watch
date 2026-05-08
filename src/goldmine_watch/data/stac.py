"""STAC client for downloading Sentinel-2 scenes.

Single-scene download for Milestone 5. Composite download for Milestone 6.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from goldmine_watch.data.cloud_mask import create_cloud_mask
from goldmine_watch.data.copernicus import (
    download_scene,
    get_access_token,
    search_scenes,
)


def _require_credentials(
    client_id: str | None, client_secret: str | None
) -> tuple[str, str]:
    """Return Copernicus credentials from arguments or environment variables."""
    client_id = client_id or os.environ.get("COPERNICUS_CLIENT_ID")
    client_secret = client_secret or os.environ.get("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Copernicus credentials missing. Set COPERNICUS_CLIENT_ID and "
            "COPERNICUS_CLIENT_SECRET environment variables."
        )
    return client_id, client_secret


def download_one_scene(
    bbox: tuple[float, float, float, float],
    date: str,
    output_path: str | Path,
    bands: list[str] | None = None,
    max_cloud_cover: float = 20.0,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> Path:
    """Download a single Sentinel-2 scene from Copernicus Data Space.

    Args:
        bbox: Bounding box as (min_x, min_y, max_x, max_y) in EPSG:4326.
        date: Date or date range string (e.g. "2023-01-01/2023-01-31").
        output_path: Where to save the output GeoTIFF.
        bands: List of band names to retrieve. Defaults to RGB + NIR + SWIR.
        max_cloud_cover: Maximum allowed cloud cover percentage.
        client_id: Copernicus OAuth Client ID (falls back to env var).
        client_secret: Copernicus OAuth Client Secret (falls back to env var).

    Returns:
        Path to the saved GeoTIFF.
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

    # Ensure SCL is always included for cloud masking
    if "SCL" not in bands:
        bands = bands + ["SCL"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client_id, client_secret = _require_credentials(client_id, client_secret)
    token = get_access_token(client_id, client_secret)

    scenes = search_scenes(bbox, date, token, max_cloud_cover=max_cloud_cover)
    if not scenes:
        raise RuntimeError(f"No scenes found for bbox={bbox} date={date}")

    item = scenes[0]
    cloud_cover = item.get("properties", {}).get("eo:cloud_cover", "unknown")
    print(f"Found scene: {item['id']} (cloud cover: {cloud_cover}%)")

    saved_path = download_scene(
        item, token, output_path, bands=bands, bbox=bbox
    )
    print(f"Saved to {saved_path}")
    return saved_path


def _composite_scenes(stack: xr.DataArray, aggregator: str) -> xr.DataArray:
    """Compute median or mean along the time axis.

    Args:
        stack: xarray DataArray with dimensions (time, band, y, x).
        aggregator: "median" or "mean".

    Returns:
        Composite DataArray with dimensions (band, y, x).

    Raises:
        ValueError: If aggregator is not "median" or "mean".
    """
    if aggregator not in {"median", "mean"}:
        raise ValueError(f"aggregator must be 'median' or 'mean', got {aggregator!r}")
    if aggregator == "median":
        return stack.median(dim="time", keep_attrs=True)  # type: ignore[no-any-return]
    return stack.mean(dim="time", keep_attrs=True)  # type: ignore[no-any-return]


def download_composite(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    output_path: str | Path,
    bands: list[str] | None = None,
    max_cloud_cover: float = 20.0,
    aggregator: str = "median",
    client_id: str | None = None,
    client_secret: str | None = None,
) -> Path:
    """Download multiple scenes and composite them.

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat].
        start_date: ISO date string (e.g. "2023-01-01").
        end_date: ISO date string (e.g. "2023-03-31").
        output_path: Where to save the composite GeoTIFF.
        bands: Band names to retrieve. Defaults to RGB + NIR + SWIR + SCL.
        max_cloud_cover: Per-scene cloud threshold.
        aggregator: "median" or "mean".
        client_id: Copernicus OAuth Client ID (falls back to env var).
        client_secret: Copernicus OAuth Client Secret (falls back to env var).

    Returns:
        Path to saved composite GeoTIFF.
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

    # Ensure SCL is always included for cloud masking
    if "SCL" not in bands:
        bands = bands + ["SCL"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client_id, client_secret = _require_credentials(client_id, client_secret)
    token = get_access_token(client_id, client_secret)

    date_range = f"{start_date}/{end_date}"
    scenes = search_scenes(
        bbox, date_range, token, max_cloud_cover=max_cloud_cover
    )

    if not scenes:
        raise RuntimeError(f"No scenes found for bbox={bbox} date={date_range}")

    print(f"Found {len(scenes)} scenes in date range")
    print(f"After cloud filter: {len(scenes)} scenes usable")

    tmp_dir = Path(tempfile.mkdtemp(prefix="composite_"))
    scene_paths: list[Path] = []

    try:
        for item in scenes:
            scene_id = item["id"]
            properties = item.get("properties", {})
            datetime_str = properties.get("datetime", "")
            if datetime_str:
                dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
                time_range = {
                    "from": dt.strftime("%Y-%m-%dT00:00:00Z"),
                    "to": dt.strftime("%Y-%m-%dT23:59:59Z"),
                }
            else:
                time_range = None

            scene_path = tmp_dir / f"{scene_id}.tif"
            download_scene(
                item,
                token,
                scene_path,
                bands=bands,
                bbox=bbox,
                time_range=time_range,
            )
            scene_paths.append(scene_path)
            print(f"  Downloaded {scene_id}")

        # Read all scenes into a stack
        refs = [rasterio.open(p) for p in scene_paths]
        try:
            data = np.stack(
                [src.read().astype(np.float32) for src in refs], axis=0
            )  # (time, band, y, x)
            profile = refs[0].profile
            transform = refs[0].transform
            crs = refs[0].crs
        finally:
            for src in refs:
                src.close()

        # Separate spectral bands from SCL
        spectral_band_names = [b for b in bands if b != "SCL"]
        scl_index = bands.index("SCL")
        spectral_indices = [i for i, b in enumerate(bands) if b != "SCL"]

        spectral_data = data[:, spectral_indices, :, :]  # (time, spectral, y, x)
        scl_data = data[:, scl_index, :, :].astype(np.uint8)  # (time, y, x)

        # Build per-scene cloud masks from SCL
        cloud_masks = np.stack(
            [create_cloud_mask(scl_data[t]) for t in range(scl_data.shape[0])],
            axis=0,
        )  # (time, y, x), 1 = clear, 0 = cloud

        before_cloud_pct = 100.0 * (1.0 - cloud_masks.mean())

        # Mask cloudy pixels with NaN so they don't influence the composite
        mask_3d = np.expand_dims(cloud_masks, axis=1)  # (time, 1, y, x)
        spectral_data[mask_3d == 0] = np.nan

        spectral_masked = xr.DataArray(
            spectral_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": range(spectral_data.shape[0]),
                "band": spectral_band_names,
            },
        )

        print(f"Computing {aggregator} composite...")
        composite = _composite_scenes(spectral_masked, aggregator)

        # After aggregation, pixels cloudy in *all* scenes are NaN
        composite_clear = (~np.isnan(composite.values[0])).astype(np.uint8)
        after_cloud_pct = 100.0 * (1.0 - composite_clear.mean())
        print(
            f"Cloud pixels reduced from {before_cloud_pct:.0f}% → {after_cloud_pct:.0f}%"
        )

        # Write to GeoTIFF (fill any remaining NaN with 0)
        composite_filled = composite.fillna(0).values.astype(
            np.float32
        )  # (bands, y, x)

        out_profile = profile.copy()
        out_profile.update(
            count=composite_filled.shape[0],
            dtype=composite_filled.dtype,
            driver="GTiff",
            transform=transform,
            crs=crs,
        )

        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(composite_filled)

        print(f"Saved to {output_path}")
    finally:
        # Cleanup temp files
        for p in scene_paths:
            if p.exists():
                p.unlink()
        if tmp_dir.exists():
            tmp_dir.rmdir()

    return output_path
