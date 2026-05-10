"""Batch inference engine for Feature 13.

Run inference on all Sentinel-2 tiles covering French Guiana using the
trained model.  Tiles are read from the local cache — no downloads are
triggered during inference.
"""

from __future__ import annotations

import time
from pathlib import Path

import rasterio
import torch

from goldmine_watch.data.tile_cache import TileCache
from goldmine_watch.data.tile_registry import TileRegistry
from goldmine_watch.inference.predict_big import predict_big_image

# Default French Guiana Sentinel-2 tile IDs
DEFAULT_TILES = ["T21NZE", "T21NZF", "T21NZG", "T22NBL", "T22NBM"]


def _auto_device() -> str:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_tile_cache_dir(cache_dir: str) -> Path:
    """Resolve the directory that actually holds cached ``.tif`` files.

    If *cache_dir* contains a ``tiles/`` sub-directory we prefer that,
    otherwise we return *cache_dir* itself.
    """
    path = Path(cache_dir)
    tiles_subdir = path / "tiles"
    if tiles_subdir.is_dir():
        return tiles_subdir
    return path


def _find_cached_tile_path(
    registry: TileRegistry, tile_id: str
) -> Path | None:
    """Return the newest valid cached GeoTIFF for *tile_id*, or None."""
    record = registry.get_tile(tile_id)
    if record is None:
        return None
    path = Path(record["filepath"])
    try:
        with rasterio.open(path) as src:
            if src.count > 0 and src.width > 0 and src.height > 0:
                return path
    except Exception:
        pass
    # Stale record — file missing or corrupted
    registry.delete_tile(record["tile_id"], record["date"])
    return None


def inference_batch(
    model_path: str = "models/phase2_best.pth",
    tile_list: list[str] | None = None,
    cache_dir: str = "data/cache",
    db_path: str = "data/cache/tiles.db",
    output_dir: str = "outputs/phase2",
    threshold: float = 0.2,
    tile_size: int = 256,
    overlap: int = 64,
    device: str | None = None,
) -> list[Path]:
    """Run batch inference on all tiles.

    Steps:
    1. Load trained model
    2. For each tile:
       - Check registry (reuse if available)
       - Run predict_big_image()
       - Save probability raster
    3. Return list of prediction paths

    Args:
        model_path: Path to the trained model checkpoint.
        tile_list: List of Sentinel-2 tile IDs to process.  Defaults to the
            five French Guiana tiles.
        cache_dir: Root cache directory.  The function automatically looks
            in a ``tiles/`` sub-directory if it exists.
        db_path: Path to the SQLite tile registry database.
        output_dir: Directory where probability GeoTIFFs are written.
        threshold: Probability threshold (stored for downstream use but not
            applied during inference).
        tile_size: Sliding-window tile size in pixels.
        overlap: Overlap between adjacent tiles in pixels.
        device: PyTorch device string (``"cuda"``, ``"mps"``, ``"cpu"``).
            ``"auto"`` is also accepted and will be resolved automatically.

    Returns:
        List of paths to the saved prediction rasters.

    Raises:
        FileNotFoundError: If the model checkpoint does not exist.
        RuntimeError: If no cached tiles can be found for the requested IDs.
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    tiles = list(tile_list) if tile_list is not None else list(DEFAULT_TILES)
    if not tiles:
        raise ValueError("No tiles specified for inference.")

    registry = TileRegistry(db_path)

    # Resolve device
    device_str = _auto_device() if device is None or device == "auto" else device
    _ = torch.device(device_str)  # validates the string

    # Infer in_channels from the first available cached tile
    first_tile_path: Path | None = None
    in_channels = 0
    for tile_id in tiles:
        path = _find_cached_tile_path(registry, tile_id)
        if path is not None:
            first_tile_path = path
            with rasterio.open(path) as src:
                in_channels = src.count
            break

    if first_tile_path is None:
        raise RuntimeError(
            f"No cached tiles found for tile IDs: {tiles}. "
            f"Checked in registry: {db_path}"
        )

    print("Batch Inference")
    print("===============")
    print(f"Loading model: {model_path}")
    print(f"Device: {device_str}")
    print("\nProcessing tiles:")

    results: list[Path] = []
    total_start = time.time()

    for i, tile_id in enumerate(tiles, start=1):
        tile_path = _find_cached_tile_path(registry, tile_id)
        if tile_path is None:
            print(f"  [{i}/{len(tiles)}] {tile_id}: NOT FOUND in cache — skipping")
            continue

        with rasterio.open(tile_path) as src:
            h, w = src.height, src.width

        out_path = output_dir_obj / f"{tile_id}_prediction.tif"
        print(
            f"  [{i}/{len(tiles)}] {tile_id}: {w}x{h} px -> "
            f"predict_big_image() -> saved"
        )

        predict_big_image(
            image_path=tile_path,
            model_path=model_path,
            output_path=out_path,
            tile_size=tile_size,
            overlap=overlap,
            in_channels=in_channels,
            device=device_str,
        )
        results.append(out_path)

    total_elapsed = time.time() - total_start
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"\nAll predictions saved to {output_dir_obj}/")
    print(f"Total inference time: {mins} minutes {secs} seconds")

    return results
