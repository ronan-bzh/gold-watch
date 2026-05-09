# Feature 13: Batch Inference Engine

**Goal:** Run inference on all Sentinel-2 tiles covering French Guiana using the trained model.

**Prerequisite:** Feature 12 (Full Territory Training) must be complete.

---

## What You Build

### Source Code

`scripts/inference_batch.py` — New script:

```python
def inference_batch(
    model_path: str = "models/phase2_best.pth",
    tile_list: list[str] | None = None,
    cache_dir: str = "data/cache",
    output_dir: str = "outputs/phase2",
    threshold: float = 0.2,
) -> list[Path]:
    """Run batch inference on all tiles.
    
    Steps:
    1. Load trained model
    2. For each tile:
       - Check cache (reuse if available)
       - Run predict_big_image()
       - Save probability raster
    3. Return list of prediction paths
    """
```

### Unit Tests

`tests/unit/test_inference_batch.py`:

```python
class TestInferenceBatch:
    def test_model_loads_successfully(self):
        """Should load phase2_best.pth without error."""
    
    def test_prediction_raster_created(self, tmp_path):
        """Should create a GeoTIFF prediction file."""
    
    def test_prediction_has_correct_shape(self, tmp_path):
        """Prediction should match input image dimensions."""
    
    def test_probability_range(self, tmp_path):
        """All values should be between 0.0 and 1.0."""
```

### Functional Tests

`tests/functional/test_feature_13_inference.py`:

```python
class TestFeature13InferenceFlow:
    def test_inference_on_single_tile(self):
        """Run inference on one cached tile."""
    
    def test_inference_on_all_tiles(self):
        """Run inference on all 5 tiles."""
    
    def test_cache_reuse_during_inference(self):
        """Should reuse cached tiles, not re-download."""
    
    def test_predictions_are_probability_heatmaps(self):
        """Output should be float32 [0,1] heatmaps, not binary masks."""
    
    def test_inference_produces_nonzero_predictions(self):
        """At least some pixels should have high probability."""
```

### Demo Script

`scripts/demo_feature13_inference.py`:

```bash
# Run batch inference
python scripts/demo_feature13_inference.py \
  --model models/phase2_best.pth \
  --output outputs/phase2
```

Output:
```
Batch Inference
===============
Loading model: models/phase2_best.pth
Device: mps

Processing tiles:
  [1/5] T21NZE: 10980x10980 px -> predict_big_image() -> saved
  [2/5] T21NZF: 10980x10980 px -> predict_big_image() -> saved
  [3/5] T21NZG: 10980x10980 px -> predict_big_image() -> saved
  [4/5] T22NBL: 10980x10980 px -> predict_big_image() -> saved
  [5/5] T22NBM: 10980x10980 px -> predict_big_image() -> saved

All predictions saved to outputs/phase2/
Total inference time: 45 minutes
```

---

## Configuration Updates

Update `configs/mvp.yaml`:

```yaml
inference:
  model_path: "models/phase2_best.pth"
  tile_size: 256
  overlap: 64
  threshold: 0.2
  output_dir: "outputs/phase2"
```

---

## Success Criteria

1. `pytest tests/unit/test_inference_batch.py -v` → **4 passed**
2. Inference completes on all tiles without crash
3. Each tile produces a valid probability GeoTIFF
4. All probability values in [0.0, 1.0]
5. Cache is reused (no re-downloads)
6. Inference time <1 hour total on GPU/MPS

---

## What You Learn

- Batch processing large geospatial datasets
- Memory management for big rasters
- Tile-based inference strategies

---

## What You DON'T Build

- Mosaic stitching
- Polygonization
- Web visualization

**Time estimate:** 3–4 hours (mostly inference time)

---

## Notes

- Inference is the slowest step. A 10,980×10,980 tile takes ~10-15 min on MPS.
- 5 tiles = ~1 hour total.
- Save predictions as float32 GeoTIFFs for later thresholding.
- Use cache aggressively — inference should never trigger downloads.
- Real-data tests require trained model from Feature 12.
