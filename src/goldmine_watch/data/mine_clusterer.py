"""Mine clustering by Sentinel-2 tile.

Groups mining polygons into Sentinel-2 tile clusters using MGRS grid references.
French Guiana spans UTM zones 21 and 22, so mines may map to tiles in either zone.
"""

from __future__ import annotations

import re
import warnings

import geopandas as gpd
import mgrs
import numpy as np
from pyproj import Geod

_SENTINEL2_TILE_RE = re.compile(r"^T\d{2}[A-Z]{3}$")
_MGRS = mgrs.MGRS()


def load_mines(geojson_path: str = "data/french_guiana_mines.geojson") -> gpd.GeoDataFrame:
    """Load all mining polygons from a GeoJSON file.

    Args:
        geojson_path: Path to the GeoJSON file.

    Returns:
        A GeoDataFrame with a ``geometry`` column.
    """
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"No features found in {geojson_path}")
    if "geometry" not in gdf.columns:
        raise ValueError(f"No geometry column in {geojson_path}")
    return gdf


def _centroids_4326(mines_gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Return centroids in EPSG:4326, projecting via RGFG95 to avoid geographic warnings."""
    # Project to RGFG95 (EPSG:2972) for a physically-meaningful centroid, then
    # convert back to WGS84 for MGRS conversion.
    projected = mines_gdf.to_crs("EPSG:2972")
    centroids = projected.geometry.centroid
    return gpd.GeoSeries(centroids, crs="EPSG:2972").to_crs("EPSG:4326")


def cluster_mines_by_tile(
    mines_gdf: gpd.GeoDataFrame,
    utm_zone: int = 22,
) -> dict[str, gpd.GeoDataFrame]:
    """Group mines by Sentinel-2 tile ID.

    Each mine is assigned to the tile that contains its centroid.

    Args:
        mines_gdf: GeoDataFrame of mining polygons. Any CRS is accepted; it will
            be re-projected internally.
        utm_zone: Unused. Kept for API compatibility with the feature spec.

    Returns:
        Mapping of tile_id (e.g. ``T21NZF``) -> GeoDataFrame of mines in that tile.
    """
    del utm_zone  # unused – French Guiana spans zones 21 and 22

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS", UserWarning)
        centroids = _centroids_4326(mines_gdf)

    tile_ids: list[str] = []
    for geom in centroids:
        mgrs_code = _MGRS.toMGRS(geom.y, geom.x, MGRSPrecision=0)
        tile_ids.append("T" + mgrs_code)

    mines_gdf = mines_gdf.copy()
    mines_gdf["tile_id"] = tile_ids

    clusters: dict[str, gpd.GeoDataFrame] = {}
    for tile_id, group in mines_gdf.groupby("tile_id", sort=False):
        clusters[tile_id] = group.drop(columns=["tile_id"]).reset_index(drop=True)

    # Return in deterministic order
    return {k: clusters[k] for k in sorted(clusters)}


def get_tile_bbox(tile_id: str) -> tuple[float, float, float, float]:
    """Return WGS84 bbox for a given Sentinel-2 tile ID.

    Args:
        tile_id: Sentinel-2 tile ID such as ``T21NZF``.

    Returns:
        ``(min_lon, min_lat, max_lon, max_lat)``.
    """
    if not _SENTINEL2_TILE_RE.match(tile_id):
        raise ValueError(f"Invalid Sentinel-2 tile ID: {tile_id}")

    mgrs_id = tile_id.lstrip("T")
    lat_min, lon_min = _MGRS.toLatLon(mgrs_id + "0000000000")
    lat_max, lon_max = _MGRS.toLatLon(mgrs_id + "9999999999")
    return (lon_min, lat_min, lon_max, lat_max)


def get_required_tiles(mines_gdf: gpd.GeoDataFrame) -> list[str]:
    """Return sorted list of unique Sentinel-2 tile IDs needed.

    Args:
        mines_gdf: GeoDataFrame of mining polygons.

    Returns:
        Sorted tile IDs.
    """
    clusters = cluster_mines_by_tile(mines_gdf)
    return list(clusters.keys())


def _union_bbox(tile_ids: list[str]) -> tuple[float, float, float, float]:
    """Compute the union bounding box of multiple tile IDs."""
    bboxes = np.array([get_tile_bbox(t) for t in tile_ids])
    return (
        float(bboxes[:, 0].min()),
        float(bboxes[:, 1].min()),
        float(bboxes[:, 2].max()),
        float(bboxes[:, 3].max()),
    )


def compute_coverage_km(tile_ids: list[str]) -> tuple[float, float]:
    """Compute approximate width and height in kilometres of a set of tiles.

    Uses geodesic distances along the bottom and left edges of the union bbox.
    """
    if not tile_ids:
        return (0.0, 0.0)
    min_lon, min_lat, max_lon, max_lat = _union_bbox(tile_ids)
    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(min_lon, min_lat, max_lon, min_lat)
    _, _, height_m = geod.inv(min_lon, min_lat, min_lon, max_lat)
    return abs(width_m) / 1000.0, abs(height_m) / 1000.0
