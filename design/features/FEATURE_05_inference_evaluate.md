# Feature 5: Single-Scene Inference & Evaluation

**Goal:** Run the trained model on the full real image and compare predictions to ground truth. Compute pixel-wise IoU and F1 on the held-out area.

**Prerequisites:** Feature 4 (trained model with saved checkpoint).

---

## What You Build

### Source Code

Update `src/goldmine_watch/inference/predict_big.py`:
- Add `--evaluate` flag that requires a labels file
- Compute pixel-wise metrics against full-resolution ground truth mask

New function in `src/goldmine_watch/inference/evaluate.py`:

```python
def evaluate_prediction(
    pred_raster_path: Path,
    labels_path: Path,
    threshold: float = 0.5,
) -> dict:
    """Compute IoU, F1, precision, recall against full-image labels.
    
    Returns dict with metrics.
    """
```

### Tests

`tests/unit/test_evaluate.py`:

```python
class TestEvaluatePrediction:
    def test_perfect_match_metrics(self, tmp_path):
        """Prediction exactly matches labels → IoU=1.0, F1=1.0."""
        
    def test_no_overlap_metrics(self, tmp_path):
        """Prediction and labels disjoint → IoU=0.0, F1=0.0."""
        
    def test_threshold_affects_metrics(self, tmp_path):
        """Lower threshold should increase recall, decrease precision."""
```

### Functional Tests

`tests/functional/test_feature_5_inference.py`:

```python
class TestFeature5InferenceFlow:
    def test_full_inference_produces_geotiff(self, tmp_path):
        """Running inference on a big image produces a valid GeoTIFF."""

    def test_tiling_creates_expected_number_of_tiles(self, tmp_path):
        """A 512x512 image with 256 tiles and 64 overlap yields 9 tiles."""

    def test_inference_on_small_image(self, tmp_path):
        """A 256x256 image with tile_size=256 produces exactly one tile."""

    def test_probability_range_0_to_1(self, tmp_path):
        """Output raster values are in [0, 1]."""

    def test_different_tile_sizes_produce_same_spatial_extent(self, tmp_path):
        """tile_size=128 and tile_size=256 both produce 512x512 outputs."""
```

### Demo Script

`scripts/demo_feature5_inference.py`:

```bash
python scripts/demo_feature5_inference.py \
  data/raw/sentinel2_scene.tif \
  models/best_model.pth \
  data/raw/mining_surfaces.gpkg \
  --threshold 0.5
```

Outputs:
- `outputs/demo/inference_comparison.png` — 3-panel figure:
  - Panel 1: Original RGB image
  - Panel 2: Ground truth mask (green)
  - Panel 3: Prediction overlay (red = predicted, yellow = overlap)
- `outputs/demo/inference_metrics.json`:
  ```json
  {
    "threshold": 0.5,
    "iou": 0.52,
    "f1": 0.68,
    "precision": 0.61,
    "recall": 0.78
  }
  ```

Console output:
```
Running inference on 10980x10980 image...
Tiling: 1936 tiles, 256px, 64px overlap
Blending complete.
Evaluating against ground truth...
IoU: 0.52 | F1: 0.68 | Precision: 0.61 | Recall: 0.78
Saved comparison to outputs/demo/inference_comparison.png
```

---

## Success Criteria

1. `pytest tests/unit/test_evaluate.py -v` → **3 passed**
2. Demo completes inference on a 10k×10k image in < 5 minutes (GPU) or < 30 minutes (CPU)
3. Output PNG clearly shows where predictions match labels (yellow) vs. miss (green only) vs. false positive (red only)
4. Metrics JSON is valid and all values are between 0.0 and 1.0
5. IoU > 0.40 (minimum viable); IoU > 0.55 (good)

---

## What You Learn

- Whether the model generalizes from patches to the full image
- Where the model makes mistakes (false positives, missed areas)
- Whether your threshold is appropriate

---

## What You DON'T Build

- Post-processing to polygons
- Temporal compositing
- QGIS project export

**Time estimate:** 2–3 hours
