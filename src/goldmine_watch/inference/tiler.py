"""Sliding-window tiling for large raster inference.

Milestone 9: Can I predict on a big image?
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def tile_image(
    image_path: str | Path,
    tile_size: int = 256,
    overlap: int = 64,
) -> list[tuple[np.ndarray, Window]]:
    """Split a large image into overlapping tiles.

    Args:
        image_path: Path to the source GeoTIFF.
        tile_size: Size of each square tile in pixels.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        List of (tile_array, rasterio_window) tuples.
        tile_array shape: (bands, tile_size, tile_size)
    """
    image_path = Path(image_path)
    stride = tile_size - overlap

    tiles = []
    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Handle edge cases where tile would exceed image bounds
                win_width = min(tile_size, width - x)
                win_height = min(tile_size, height - y)
                window = Window(x, y, win_width, win_height)
                tile = src.read(window=window)

                # Pad if necessary
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    pad_y = tile_size - tile.shape[1]
                    pad_x = tile_size - tile.shape[2]
                    tile = np.pad(tile, ((0, 0), (0, pad_y), (0, pad_x)), mode="constant")

                tiles.append((tile, window))

    return tiles
