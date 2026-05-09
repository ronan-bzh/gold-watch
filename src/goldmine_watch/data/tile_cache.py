"""Cache manager for Sentinel-2 tiles.

Provides a filesystem-based cache for Sentinel-2 scenes so that tiles are
downloaded once and reused across training and inference runs.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import mgrs
import rasterio

from goldmine_watch.data.copernicus import (
    download_scene,
    get_access_token,
    search_scenes,
)


def _require_credentials() -> tuple[str, str]:
    """Return Copernicus credentials from environment variables."""
    client_id = os.environ.get("COPERNICUS_CLIENT_ID")
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Copernicus credentials missing. Set COPERNICUS_CLIENT_ID and "
            "COPERNICUS_CLIENT_SECRET environment variables. "
            "See the project setup docs for instructions on obtaining credentials."
        )
    return client_id, client_secret


def _is_valid_geotiff(path: Path) -> bool:
    """Return True if *path* points to a readable, non-empty GeoTIFF."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return bool(src.count > 0 and src.width > 0 and src.height > 0)
    except Exception:
        return False


def _scene_date(item: dict) -> str:
    """Extract acquisition date from a STAC item and format as YYYYMMDD."""
    dt_str = item.get("properties", {}).get("datetime", "")
    if not dt_str:
        raise ValueError("STAC item missing datetime property")
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d")


class TileCache:
    """Filesystem cache for Sentinel-2 tiles.

    Tiles are stored as ``<tile_id>_<date>.tif`` under *cache_dir*.
    The first request for a tile searches the Copernicus STAC catalog,
    downloads the best-matching scene, and caches it. Subsequent requests
    return the cached path immediately.
    """

    def __init__(self, cache_dir: str = "data/cache/tiles"):
        """Initialize cache directory.

        Args:
            cache_dir: Root directory for cached tiles. Created if missing.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_tile(
        self,
        tile_id: str,
        date_range: str,
        bbox: tuple[float, float, float, float],
        bands: list[str] | None = None,
    ) -> Path:
        """Return cached tile path or download if missing.

        Checks local cache first. Any existing valid GeoTIFF matching
        ``<tile_id>_*.tif`` is returned immediately. Corrupted files are
        removed during the scan. Only when no cached tile is found does the
        method query the Copernicus STAC catalog and download the
        best-matching scene.

        Args:
            tile_id: Sentinel-2 tile ID (e.g. ``T21NZF``).
            date_range: Date or date range string (e.g. ``2023-01-01/2023-01-31``).
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
            bands: List of band names to retrieve. Defaults to the standard
                RGB + NIR + SWIR + SCL set.

        Returns:
            Path to the cached GeoTIFF.

        Raises:
            RuntimeError: If credentials are missing or no scenes match the query.
        """
        if bands is None:
            bands = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

        # Ensure SCL is always included for cloud masking
        if "SCL" not in bands:
            bands = bands + ["SCL"]

        # 1. Cache-first lookup — any valid tile for this ID is acceptable.
        #    Corrupted files are cleaned up as we scan.
        cached = sorted(self.cache_dir.glob(f"{tile_id}_*.tif"), reverse=True)
        for path in cached:
            if _is_valid_geotiff(path):
                return path
            # Corrupted — remove so it doesn't shadow valid tiles
            path.unlink(missing_ok=True)

        # 2. Not in cache — authenticate, search, and download
        client_id, client_secret = _require_credentials()
        token = get_access_token(client_id, client_secret)

        scenes = search_scenes(bbox, date_range, token)
        if not scenes:
            raise RuntimeError(f"No scenes found for bbox={bbox} date={date_range}")

        best_scene = scenes[0]
        date_str = _scene_date(best_scene)
        cache_path = self.cache_dir / f"{tile_id}_{date_str}.tif"

        # Remove any existing corrupted file with the same name
        if cache_path.exists():
            cache_path.unlink()

        # Download to a temp file and rename atomically to avoid race
        # conditions when multiple processes cache the same tile.
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".tif", prefix=f"{tile_id}_{date_str}_", dir=str(self.cache_dir)
        )
        os.close(tmp_fd)
        tmp_file = Path(tmp_path)
        try:
            download_scene(best_scene, token, tmp_file, bands=bands, bbox=bbox)
            tmp_file.rename(cache_path)
        except Exception:
            tmp_file.unlink(missing_ok=True)
            raise

        return cache_path

    def list_cached_tiles(self) -> list[Path]:
        """List all tiles currently in cache.

        Returns:
            Sorted list of paths to cached ``.tif`` files.
        """
        return sorted(self.cache_dir.glob("*.tif"))

    def get_cache_size_mb(self) -> float:
        """Return total cache size in megabytes.

        Returns:
            Sum of all cached file sizes, in MB.
        """
        total_bytes = sum(p.stat().st_size for p in self.cache_dir.glob("*.tif"))
        return total_bytes / (1024 * 1024)

    def clear_cache(self) -> None:
        """Remove all cached tiles."""
        for path in self.cache_dir.glob("*.tif"):
            path.unlink(missing_ok=True)


def get_tile_id_from_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Determine Sentinel-2 tile ID (e.g. ``T21NZF``) from a WGS84 bbox.

    Uses the MGRS (Military Grid Reference System) convention: the tile ID
    is ``T`` + UTM zone (2 digits) + latitude band (1 letter) +
    100km grid square (2 letters).

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.

    Returns:
        Sentinel-2 tile ID string.

    Raises:
        ValueError: If the bbox is outside valid geographic bounds or
            outside Sentinel-2 MGRS coverage.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    if not (-180.0 <= min_lon <= max_lon <= 180.0):
        raise ValueError("Invalid longitude range in bbox")
    if not (-90.0 <= min_lat <= max_lat <= 90.0):
        raise ValueError("Invalid latitude range in bbox")

    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    # Sentinel-2 MGRS coverage is limited to -80° to +84° latitude
    if center_lat < -80.0 or center_lat > 84.0:
        raise ValueError(
            f"Center latitude {center_lat:.2f} outside Sentinel-2 MGRS coverage " "(-80° to +84°)"
        )

    m = mgrs.MGRS()
    mgrs_str = m.toMGRS(center_lat, center_lon)

    # mgrs_str format: ZZBLLNNNNNNNNNN (e.g. "21NZG2969205434")
    # ZZ = UTM zone (01-60)
    # B  = latitude band (C-X)
    # LL = 100km grid square (2 letters)
    zone = mgrs_str[:2]
    band = mgrs_str[2]
    square = mgrs_str[3:5]
    return f"T{zone}{band}{square}"
