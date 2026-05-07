"""STAC client for downloading Sentinel-2 scenes.

Single-scene download for Milestone 5. No retry, no fallback, no compositing.
"""

from pathlib import Path

import planetary_computer
import pystac_client
import rasterio
import stackstac
from rasterio.crs import CRS


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
        bands = ["B02", "B03", "B04", "B08", "B11", "B12"]

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

    print(f"Saved to {output_path}")
    return output_path
