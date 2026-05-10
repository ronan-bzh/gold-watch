"""Build complete training dataset from all mines across multiple Sentinel-2 tiles.

Orchestrates clustering, tile download (cache-first), mine-centered patch
extraction, background patch extraction, and spatial train/val splitting.
"""

from __future__ import annotations

import hashlib
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window

from goldmine_watch.data.ingest import burn_mask
from goldmine_watch.data.mine_clusterer import (
    cluster_mines_by_tile,
    get_tile_bbox,
    load_mines,
)
from goldmine_watch.data.tile_cache import TileCache

logger = logging.getLogger(__name__)


def _extract_mine_centered_patches(
    image_path: Path,
    mask: np.ndarray,
    mines_gdf: gpd.GeoDataFrame,
    patch_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract patches centered on each mine centroid.

    Patches are clamped to image bounds, so mines near edges may not be
    exactly centered but will still be contained in the patch.

    Args:
        image_path: Path to the tile GeoTIFF.
        mask: Binary mask of shape (height, width) with 1 for mine pixels.
        mines_gdf: GeoDataFrame of mines in the tile's CRS.
        patch_size: Size of each square patch in pixels.

    Returns:
        List of (image_patch, mask_patch) tuples.
    """
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    half = patch_size // 2

    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

        # Reproject mines to raster CRS if needed
        if mines_gdf.crs is not None and crs is not None and not mines_gdf.crs.equals(crs):
            mines_gdf = mines_gdf.to_crs(crs)

        for geom in mines_gdf.geometry:
            if geom.is_empty:
                continue
            cx, cy = geom.centroid.x, geom.centroid.y
            row, col = rowcol(transform, cx, cy)

            x = int(col) - half
            y = int(row) - half

            # Clamp to image bounds
            # Guard against images smaller than patch_size
            if width < patch_size or height < patch_size:
                logger.warning(
                    "Image %s (%dx%d) is smaller than patch_size (%d). Skipping mine patch.",
                    image_path,
                    width,
                    height,
                    patch_size,
                )
                continue

            x = max(0, min(x, width - patch_size))
            y = max(0, min(y, height - patch_size))

            window = Window(x, y, patch_size, patch_size)
            image_patch = src.read(window=window)
            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            patches.append((image_patch, mask_patch))

    return patches


def _extract_background_patches(
    image_path: Path,
    mask: np.ndarray,
    patch_size: int,
    num_patches: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract random background patches that contain no mine pixels.

    Args:
        image_path: Path to the tile GeoTIFF.
        mask: Binary mask of shape (height, width) with 1 for mine pixels.
        patch_size: Size of each square patch in pixels.
        num_patches: Number of background patches to extract.
        random_seed: Seed for reproducible random sampling.

    Returns:
        List of (image_patch, mask_patch) tuples.
    """
    rng = random.Random(random_seed)
    patches: list[tuple[np.ndarray, np.ndarray]] = []

    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width

        max_x = width - patch_size
        max_y = height - patch_size

        if max_x < 0 or max_y < 0:
            logger.warning(
                "Image %s is smaller than patch_size (%d). No background patches extracted.",
                image_path,
                patch_size,
            )
            return patches

        attempts = 0
        max_attempts = num_patches * 100

        while len(patches) < num_patches and attempts < max_attempts:
            attempts += 1
            x = rng.randint(0, max_x)
            y = rng.randint(0, max_y)

            mask_patch = mask[y : y + patch_size, x : x + patch_size]
            if np.any(mask_patch > 0):
                continue  # Contains mine pixels — skip

            window = Window(x, y, patch_size, patch_size)
            image_patch = src.read(window=window)
            patches.append((image_patch, mask_patch))

    if len(patches) < num_patches:
        logger.warning(
            "Only found %d background patches (requested %d) for %s",
            len(patches),
            num_patches,
            image_path,
        )

    return patches


def _spatial_split_tiles(
    tile_ids: list[str],
    train_val_ratio: tuple[float, float],
    random_seed: int,
) -> tuple[list[str], list[str]]:
    """Split tiles into train and validation sets.

    Ensures spatial separation by assigning whole tiles to either train or val.

    Args:
        tile_ids: List of tile IDs.
        train_val_ratio: (train_fraction, val_fraction).
        random_seed: Seed for deterministic splitting.

    Returns:
        Tuple of (train_tile_ids, val_tile_ids).
    """
    if len(tile_ids) < 2:
        raise ValueError(
            f"Spatial train/val split requires at least 2 tiles, got {len(tile_ids)}. "
            "Use more tiles or disable spatial splitting."
        )

    rng = random.Random(random_seed)
    shuffled = list(tile_ids)
    rng.shuffle(shuffled)

    train_frac = train_val_ratio[0]
    n_train = max(1, int(round(len(shuffled) * train_frac)))

    # Ensure at least one tile in each split
    if n_train >= len(shuffled):
        n_train = len(shuffled) - 1
    if n_train == 0:
        n_train = 1

    return shuffled[:n_train], shuffled[n_train:]


def _save_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    start_idx: int = 0,
) -> int:
    """Save patches as standard .npy files.

    Args:
        patches: List of (image_patch, mask_patch) tuples.
        output_dir: Directory to save files.
        start_idx: Starting index for naming.

    Returns:
        Next available index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = start_idx

    for image_patch, mask_patch in patches:
        np.save(output_dir / f"image_{idx:05d}.npy", image_patch)
        np.save(output_dir / f"mask_{idx:05d}.npy", mask_patch)
        idx += 1

    return idx


def _compute_mine_bbox(
    mines_gdf: gpd.GeoDataFrame, padding_degrees: float = 0.05
) -> tuple[float, float, float, float]:
    """Compute a padded bounding box around mining polygons in WGS84.

    Using a mine-centered bbox instead of the full Sentinel-2 tile bbox
    dramatically reduces the download size and avoids Copernicus Process
    API 400 errors for oversized requests.

    Args:
        mines_gdf: GeoDataFrame with mine geometries (any CRS).
        padding_degrees: Padding in degrees around the mine extents.
            Default 0.05 deg ≈ 5.5 km at the equator.

    Returns:
        ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    """
    # Ensure we're working in WGS84 for the bbox
    if mines_gdf.crs is not None and not mines_gdf.crs.to_string() == "EPSG:4326":
        gdf_wgs84 = mines_gdf.to_crs("EPSG:4326")
    else:
        gdf_wgs84 = mines_gdf

    bounds = gdf_wgs84.total_bounds  # (minx, miny, maxx, maxy)
    min_lon = bounds[0] - padding_degrees
    min_lat = bounds[1] - padding_degrees
    max_lon = bounds[2] + padding_degrees
    max_lat = bounds[3] + padding_degrees

    # Clamp to valid WGS84 ranges
    min_lon = max(-180.0, min_lon)
    min_lat = max(-90.0, min_lat)
    max_lon = min(180.0, max_lon)
    max_lat = min(90.0, max_lat)

    return (min_lon, min_lat, max_lon, max_lat)


def build_training_dataset(
    mines_geojson: str = "data/french_guiana_mines.geojson",
    output_dir: str = "data/splits",
    num_background_per_tile: int = 100,
    patch_size: int = 256,
    date_range: str = "2023-06-01/2023-12-31",
    train_val_ratio: tuple[float, float] = (0.8, 0.2),
    random_seed: int = 42,
    cache_dir: str = "data/cache/tiles",
    db_path: str = "data/cache/tiles.db",
) -> dict[str, Any]:
    """Build complete training dataset from all mines.

    Steps:
        1. Cluster mines by Sentinel-2 tile.
        2. Download each tile (using cache).
        3. Extract mine-centered patches.
        4. Extract background patches.
        5. Split into train/val (spatial, by tile).
        6. Save patches as .npy files.

    Args:
        mines_geojson: Path to the GeoJSON file with mining polygons.
        output_dir: Directory to save train/ and val/ splits.
        num_background_per_tile: Number of background patches to extract per tile.
        patch_size: Size of each square patch in pixels.
        date_range: Date range for tile downloads (ISO 8601 interval).
        train_val_ratio: (train_fraction, val_fraction) tuple.
        random_seed: Seed for reproducible random sampling and splitting.
        cache_dir: Directory for tile cache.
        db_path: Path to the SQLite tile registry database.

    Returns:
        Statistics dictionary with keys:
        - total_patches, train_patches, val_patches
        - positive_patches, negative_patches
        - num_tiles, train_tiles, val_tiles
        - tile_names, output_dir
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"

    # Clean and recreate output subdirectories only (never delete the parent)
    for subdir in (train_dir, val_dir):
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)

    # Load and cluster mines
    logger.info("Loading mines from %s", mines_geojson)
    mines_gdf = load_mines(mines_geojson)
    logger.info("Loaded %d mines", len(mines_gdf))

    clusters = cluster_mines_by_tile(mines_gdf)
    tile_ids = sorted(clusters.keys())
    logger.info("Clustered into %d tiles: %s", len(tile_ids), ", ".join(tile_ids))

    # Spatial split
    train_tiles, val_tiles = _spatial_split_tiles(tile_ids, train_val_ratio, random_seed)
    logger.info(
        "Spatial split: %d train tiles, %d val tiles",
        len(train_tiles),
        len(val_tiles),
    )

    cache = TileCache(cache_dir, db_path=db_path)

    stats_by_tile: dict[str, dict[str, int]] = {}
    train_idx = 0
    val_idx = 0
    total_positive = 0
    total_negative = 0

    for tile_id in tile_ids:
        tile_mines = clusters[tile_id]
        # Use mine-centered bbox instead of full tile bbox to keep downloads small
        bbox = _compute_mine_bbox(tile_mines, padding_degrees=0.05)

        logger.info("[%s] Downloading tile (mine bbox: %.4f, %.4f, %.4f, %.4f)...", tile_id, *bbox)
        try:
            tile_path = cache.get_tile(
                tile_id=tile_id,
                date_range=date_range,
                bbox=bbox,
            )
        except Exception:
            logger.exception("Failed to download tile %s — skipping", tile_id)
            continue

        logger.info("[%s] Extracting patches...", tile_id)

        # Rasterize mine labels to match the tile
        mask = burn_mask(tile_mines, tile_path)

        # Mine-centered patches
        mine_patches = _extract_mine_centered_patches(tile_path, mask, tile_mines, patch_size)

        # Background patches
        bg_patches = _extract_background_patches(
            image_path=tile_path,
            mask=mask,
            patch_size=patch_size,
            num_patches=num_background_per_tile,
            random_seed=random_seed + int(hashlib.md5(tile_id.encode()).hexdigest(), 16) % 10000,
        )

        tile_positive = sum(1 for _, mp in mine_patches if np.any(mp > 0))
        tile_negative = len(bg_patches) + sum(1 for _, mp in mine_patches if not np.any(mp > 0))
        total_positive += tile_positive
        total_negative += tile_negative

        stats_by_tile[tile_id] = {
            "mines": len(tile_mines),
            "mine_patches": len(mine_patches),
            "background_patches": len(bg_patches),
            "positive_patches": tile_positive,
            "negative_patches": tile_negative,
        }

        # Save to appropriate split directory
        if tile_id in train_tiles:
            train_idx = _save_patches(mine_patches, train_dir, train_idx)
            train_idx = _save_patches(bg_patches, train_dir, train_idx)
        else:
            val_idx = _save_patches(mine_patches, val_dir, val_idx)
            val_idx = _save_patches(bg_patches, val_dir, val_idx)

    train_patches = len(list(train_dir.glob("image_*.npy")))
    val_patches = len(list(val_dir.glob("image_*.npy")))

    return {
        "total_patches": train_patches + val_patches,
        "train_patches": train_patches,
        "val_patches": val_patches,
        "positive_patches": total_positive,
        "negative_patches": total_negative,
        "num_tiles": len(tile_ids),
        "train_tiles": train_tiles,
        "val_tiles": val_tiles,
        "tile_names": tile_ids,
        "output_dir": str(output_path),
        "stats_by_tile": stats_by_tile,
    }
