"""Evaluate predictions against ground-truth labels.

Feature 5: Compute pixel-wise IoU, F1, precision, recall.
"""

from pathlib import Path

import rasterio

from goldmine_watch.data.ingest import burn_mask, load_labels
from goldmine_watch.training.metrics import compute_f1, compute_iou, compute_precision_recall


def evaluate_prediction(
    pred_raster_path: Path,
    labels_path: Path,
    threshold: float = 0.5,
) -> dict:
    """Compute IoU, F1, precision, recall against full-image labels.

    Args:
        pred_raster_path: Path to the predicted probability GeoTIFF.
        labels_path: Path to vector labels (GeoPackage, Shapefile, GeoJSON).
        threshold: Probability threshold to binarize predictions.

    Returns:
        dict with keys: threshold, iou, f1, precision, recall.
    """
    with rasterio.open(pred_raster_path) as src:
        pred = src.read(1)

    gdf = load_labels(labels_path)
    target = burn_mask(gdf, pred_raster_path)

    iou = compute_iou(pred, target, threshold=threshold)
    precision, recall = compute_precision_recall(pred, target, threshold=threshold)
    f1 = compute_f1(pred, target, threshold=threshold)

    return {
        "threshold": threshold,
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }
