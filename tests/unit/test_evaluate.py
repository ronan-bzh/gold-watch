"""Unit tests for inference evaluation."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.inference.evaluate import evaluate_prediction

TARGET_CRS = "EPSG:2972"


def _make_prob_raster(path: Path, data: np.ndarray) -> Path:
    """Save a probability array as a GeoTIFF."""
    height, width = data.shape
    transform = from_origin(200_000.0, 500_000.0 + height * 10.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _make_labels(path: Path, polygons: list[Polygon]) -> Path:
    """Save polygons as a GeoPackage."""
    gdf = gpd.GeoDataFrame(
        {"label": ["mining"] * len(polygons)},
        geometry=polygons,
        crs=TARGET_CRS,
    )
    gdf.to_file(path, driver="GPKG")
    return path


class TestEvaluatePrediction:
    """Tests for evaluate_prediction."""

    def test_perfect_match_metrics(self, tmp_path: Path) -> None:
        """Prediction exactly matches labels → IoU=1.0, F1=1.0."""
        size = 100
        pred = np.ones((size, size), dtype=np.float32) * 0.9
        pred_raster = _make_prob_raster(tmp_path / "pred.tif", pred)

        polygons = [
            Polygon(
                [
                    (200_000.0, 500_000.0),
                    (200_000.0 + size * 10.0, 500_000.0),
                    (200_000.0 + size * 10.0, 500_000.0 + size * 10.0),
                    (200_000.0, 500_000.0 + size * 10.0),
                ]
            )
        ]
        labels_path = _make_labels(tmp_path / "labels.gpkg", polygons)

        metrics = evaluate_prediction(pred_raster, labels_path, threshold=0.5)
        assert metrics["iou"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_no_overlap_metrics(self, tmp_path: Path) -> None:
        """Prediction and labels disjoint → IoU=0.0, F1=0.0."""
        size = 100
        pred = np.zeros((size, size), dtype=np.float32)
        pred_raster = _make_prob_raster(tmp_path / "pred.tif", pred)

        # Labels in the top-left corner, but prediction is all zeros
        polygons = [
            Polygon(
                [
                    (200_000.0, 500_000.0 + 50 * 10.0),
                    (200_000.0 + 50 * 10.0, 500_000.0 + 50 * 10.0),
                    (200_000.0 + 50 * 10.0, 500_000.0 + 100 * 10.0),
                    (200_000.0, 500_000.0 + 100 * 10.0),
                ]
            )
        ]
        labels_path = _make_labels(tmp_path / "labels.gpkg", polygons)

        metrics = evaluate_prediction(pred_raster, labels_path, threshold=0.5)
        assert metrics["iou"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)

    def test_threshold_affects_metrics(self, tmp_path: Path) -> None:
        """Lower threshold should increase recall, decrease precision."""
        size = 100
        # Predictions: high prob in rows 0:10 (overlaps labels),
        #              low prob in rows 10:50 (partially overlaps, partially not)
        #              zero elsewhere
        pred = np.zeros((size, size), dtype=np.float32)
        pred[0:10, :] = 0.9
        pred[10:50, :] = 0.3
        pred_raster = _make_prob_raster(tmp_path / "pred.tif", pred)

        # Labels cover rows 0:20 (full width)
        polygons = [
            Polygon(
                [
                    (200_000.0, 500_000.0 + 80 * 10.0),
                    (200_000.0 + size * 10.0, 500_000.0 + 80 * 10.0),
                    (200_000.0 + size * 10.0, 500_000.0 + 100 * 10.0),
                    (200_000.0, 500_000.0 + 100 * 10.0),
                ]
            )
        ]
        labels_path = _make_labels(tmp_path / "labels.gpkg", polygons)

        metrics_high = evaluate_prediction(pred_raster, labels_path, threshold=0.5)
        metrics_low = evaluate_prediction(pred_raster, labels_path, threshold=0.2)

        # Lower threshold should increase recall (more predicted positives catch more labels)
        assert metrics_low["recall"] > metrics_high["recall"]
        # Lower threshold should decrease precision (more false positives in rows 20:50)
        assert metrics_low["precision"] < metrics_high["precision"]
