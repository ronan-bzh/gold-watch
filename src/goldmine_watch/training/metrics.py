"""Metrics for binary semantic segmentation."""

import numpy as np


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for binary segmentation.

    Args:
        pred: Predicted logits or probabilities, any shape.
        target: Ground truth binary mask, same shape as pred.
        threshold: Threshold to binarize predictions.

    Returns:
        IoU score in [0, 1].
    """
    pred_binary = (pred >= threshold).astype(bool)
    target_binary = (target >= threshold).astype(bool)
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_precision_recall(
    pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    """Return (precision, recall) for binary segmentation.

    Args:
        pred: Predicted logits or probabilities, any shape.
        target: Ground truth binary mask, same shape as pred.
        threshold: Threshold to binarize predictions.

    Returns:
        Tuple of (precision, recall) in [0, 1].
    """
    pred_binary = (pred >= threshold).astype(bool)
    target_binary = (target >= threshold).astype(bool)
    tp = int(np.logical_and(pred_binary, target_binary).sum())
    fp = int(np.logical_and(pred_binary, ~target_binary).sum())
    fn = int(np.logical_and(~pred_binary, target_binary).sum())
    if tp + fp + fn == 0:
        return 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def compute_f1(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    Args:
        pred: Predicted logits or probabilities, any shape.
        target: Ground truth binary mask, same shape as pred.
        threshold: Threshold to binarize predictions.

    Returns:
        F1 score in [0, 1].
    """
    precision, recall = compute_precision_recall(pred, target, threshold)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
