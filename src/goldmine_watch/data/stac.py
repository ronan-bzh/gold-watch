"""STAC client for downloading Sentinel-2 scenes.

Single-scene download for Milestone 5. Composite download for Milestone 6.
"""

from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import stackstac
import xarray as xr
from rasterio.crs import CRS

from goldmine_watch.data.cloud_mask import create_cloud_mask


def download_one_scene(
    bbox: tuple[float, float, float, float],
    date: str,
    output_path: str | Path,
    bands: list[str] | None = None,
    max_cloud_cover: float = 20.0,
) -> Path:
    """Download a single Sentinel-2 scene from Microsoft Planetary Computer.

    Args:
        bbox: Bounding box as (min_x, min_y, max_x, max_y) in EPSG:4326.
        date: Date or date range string (e.g. "2023-01-01/2023-01-31").
        output_path: Where to save the output GeoTIFF.
        bands: List of band names to retrieve. Defaults to RGB + NIR + SWIR.
        max_cloud_cover: Maximum allowed cloud cover percentage.

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

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=1,
    )
    items = list(search.get_all_items())

    if not items:
        raise RuntimeError(f"No scenes found for bbox={bbox} date={date}")

    item = items[0]
    print(
        f"Found scene: {item.id} (cloud cover: {item.properties.get('eo:cloud_cover', 'unknown')}%)"
    )

    stack = stackstac.stack(
        [item],
        assets=bands,
        bounds_latlon=bbox,
        resolution=10,
        dtype="uint16",
        rescale=False,
        epsg=32622,
    )

    # Write to GeoTIFF
    data = stack.squeeze().values  # (bands, y, x)
    transform = stackstac.bounds_to_transform(stack.attrs["bounds"], stack.shape[-2:])
    crs = CRS.from_epsg(32622)  # UTM 22N — French Guiana fallback

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    # Tag the SCL band so load_scl_band can find it
    with rasterio.open(output_path, "r+") as dst:
        scl_index = bands.index("SCL") + 1
        dst.set_band_description(scl_index, "SCL")

    print(f"Saved to {output_path}")
    return output_path


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

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    date_range = f"{start_date}/{end_date}"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.get_all_items())

    if not items:
        raise RuntimeError(f"No scenes found for bbox={bbox} date={date_range}")

    print(f"Found {len(items)} scenes in date range")
    print(f"After cloud filter: {len(items)} scenes usable")

    stack = stackstac.stack(
        items,
        assets=bands,
        bounds_latlon=bbox,
        resolution=10,
        dtype="uint16",
        rescale=False,
        epsg=32622,
    )

    # Separate SCL from spectral bands
    spectral_bands = [b for b in bands if b != "SCL"]
    spectral_stack = stack.sel(band=spectral_bands)
    scl_stack = stack.sel(band=["SCL"])

    # Build per-scene cloud masks from SCL
    scl_data = scl_stack.values[:, 0, :, :].astype(np.uint8)  # (time, y, x)
    cloud_masks = np.stack(
        [create_cloud_mask(scl_data[t]) for t in range(scl_data.shape[0])],
        axis=0,
    )  # (time, y, x), 1 = clear, 0 = cloud

    before_cloud_pct = 100.0 * (1.0 - cloud_masks.mean())

    # Mask cloudy pixels with NaN so they don't influence the composite
    spectral_values = spectral_stack.values.astype(np.float32)  # (time, band, y, x)
    mask_3d = np.expand_dims(cloud_masks, axis=1)  # (time, 1, y, x)
    spectral_values[mask_3d == 0] = np.nan

    spectral_masked = xr.DataArray(
        spectral_values,
        dims=spectral_stack.dims,
        coords=spectral_stack.coords,
        attrs=spectral_stack.attrs,
    )

    print(f"Computing {aggregator} composite...")
    composite = _composite_scenes(spectral_masked, aggregator)

    # After aggregation, pixels cloudy in *all* scenes are NaN
    composite_clear = (~np.isnan(composite.values[0])).astype(np.uint8)
    after_cloud_pct = 100.0 * (1.0 - composite_clear.mean())
    print(f"Cloud pixels reduced from {before_cloud_pct:.0f}% → {after_cloud_pct:.0f}%")

    # Write to GeoTIFF (fill any remaining NaN with 0)
    data = composite.fillna(0).values.astype(np.float32)  # (bands, y, x)
    transform = stackstac.bounds_to_transform(stack.attrs["bounds"], data.shape[-2:])
    crs = CRS.from_epsg(32622)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    print(f"Saved to {output_path}")
    return output_path
