"""Unit tests for segmentation metrics."""

import numpy as np
import pytest

from goldmine_watch.training.metrics import compute_f1, compute_iou, compute_precision_recall


class TestMetrics:
    """Tests for binary segmentation metrics."""

    def test_perfect_prediction_iou_is_one(self) -> None:
        """Pred == target → IoU = 1.0."""
        pred = np.ones((10, 10), dtype=np.float32)
        target = np.ones((10, 10), dtype=np.float32)
        assert compute_iou(pred, target) == pytest.approx(1.0)

    def test_all_wrong_iou_is_zero(self) -> None:
        """Pred and target disjoint → IoU = 0.0."""
        pred = np.ones((10, 10), dtype=np.float32)
        target = np.zeros((10, 10), dtype=np.float32)
        assert compute_iou(pred, target) == pytest.approx(0.0)

    def test_f1_balances_precision_recall(self) -> None:
        """F1 is 2 * (P*R) / (P+R)."""
        # 2 TP, 1 FP, 1 FN → precision=2/3, recall=2/3, f1=2/3
        pred = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
        target = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
        precision, recall = compute_precision_recall(pred, target)
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert compute_f1(pred, target) == pytest.approx(expected_f1)

    def test_metrics_on_random(self) -> None:
        """Random predictions should give IoU ≈ 0.01–0.05 on imbalanced data."""
        rng = np.random.default_rng(42)
        # 90% negative class
        target = (rng.random((100, 100)) > 0.9).astype(np.float32)
        pred = rng.random((100, 100)).astype(np.float32)
        iou = compute_iou(pred, target)
        assert 0.0 <= iou <= 0.10
