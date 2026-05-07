"""Blending utilities for overlapping tile predictions.

Milestone 9: Average overlapping predictions.
"""

import numpy as np


def blend_predictions(
    canvas: np.ndarray,
    counts: np.ndarray,
    tile_prediction: np.ndarray,
    x: int,
    y: int,
    tile_size: int,
) -> None:
    """Add a tile prediction to the canvas and increment the overlap counter.

    Args:
        canvas: Array of shape (height, width) accumulating predictions.
        counts: Array of shape (height, width) counting overlaps per pixel.
        tile_prediction: Array of shape (tile_size, tile_size) to add.
        x: Top-left column index in the canvas.
        y: Top-left row index in the canvas.
        tile_size: Expected tile size (used to clip if edge tile).
    """
    h, w = canvas.shape
    pred_h, pred_w = tile_prediction.shape

    # Clip to canvas bounds
    max_h = min(pred_h, h - y)
    max_w = min(pred_w, w - x)

    canvas[y : y + max_h, x : x + max_w] += tile_prediction[:max_h, :max_w]
    counts[y : y + max_h, x : x + max_w] += 1


def normalize_canvas(canvas: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Divide accumulated predictions by overlap counts.

    Args:
        canvas: Accumulated prediction array.
        counts: Overlap count array.

    Returns:
        Averaged prediction array.
    """
    # Avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    return np.asarray(canvas / counts, dtype=np.float32)
