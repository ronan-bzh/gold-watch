# Feature 8: Spectral Rule-Based Baseline (Stage 1)

**Goal:** Build a simple rule-based detector using NDVI + BSI thresholds. This provides a **baseline** to beat with the AI model.

**Prerequisites:** Feature 1 (real image downloaded).

---

## What You Build

### Source Code

`src/goldmine_watch/baseline/rules.py` — New module:

```python
def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index."""

def compute_bsi(red: np.ndarray, nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Compute Bare Soil Index."""

def detect_mining_rules(
    image_path: Path,
    ndvi_threshold: float = 0.2,
    bsi_threshold: float = 0.1,
) -> np.ndarray:
    """Return binary mask where NDVI < threshold AND BSI > threshold.
    
    These rules flag bare soil with low vegetation — the spectral signature
    of cleared mining areas.
    """

def rules_to_polygons(mask: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """Convert binary rule mask to vector polygons."""
```

### Tests

`tests/unit/test_baseline.py`:

```python
class TestBaselineRules:
    def test_ndvi_range(self):
        """NDVI should be between -1 and 1."""
        
    def test_bare_soil_detected(self):
        """Pixel with low NDVI and high BSI should be flagged."""
        
    def test_forest_not_detected(self):
        """Pixel with high NDVI should NOT be flagged."""
        
    def test_rules_produce_polygons(self):
        """rules_to_polygons should return a non-empty GeoDataFrame."""
```

### Functional Tests

`tests/functional/test_feature_8_baseline.py`:

```python
class TestFeature8BaselineFlow:
    def test_detect_mining_rules_produces_mask(self, tmp_path):
        """Running rules on a multiband image produces a binary mask."""

    def test_ndvi_range_between_minus_one_and_one(self, tmp_path):
        """NDVI values should be in [-1, 1]."""

    def test_bare_soil_detected_low_ndvi_high_bsi(self, tmp_path):
        """Bare soil pixels (low NIR, high RED+SWIR) should be flagged."""

    def test_forest_not_detected_high_ndvi(self, tmp_path):
        """Vegetation pixels (high NIR, low RED) should NOT be flagged."""

    def test_rules_to_polygons_returns_geodataframe(self, tmp_path):
        """Converting a rule mask to polygons yields a GeoDataFrame."""

    def test_empty_mask_produces_empty_geodataframe(self, tmp_path):
        """All-zero mask should produce an empty GeoDataFrame."""

    def test_threshold_tuning_changes_mask(self, tmp_path):
        """Stricter threshold should produce fewer detections."""
```

### Demo Script

`scripts/demo_feature8_baseline.py`:

```bash
python scripts/demo_feature8_baseline.py \
  data/raw/sentinel2_scene.tif \
  --ndvi-threshold 0.2 \
  --bsi-threshold 0.1
```

Outputs:
- `outputs/demo/baseline_mask.png` — Binary mask from rules
- `outputs/demo/baseline_comparison.png` — Side-by-side:
  - AI model prediction (from Feature 5)
  - Rule-based prediction
  - Ground truth labels
- `outputs/demo/baseline_polygons.gpkg` — Rule-based polygons

Console output:
```
Computing NDVI and BSI...
Rule-based detections: 234 polygons
AI model detections: 89 polygons
Ground truth labels: 42 polygons
Rule IoU vs GT: 0.31
AI IoU vs GT: 0.52
✅ AI beats baseline by +67%
```

---

## Success Criteria

1. `pytest tests/unit/test_baseline.py -v` → **4 passed**
2. Demo runs in < 2 minutes (no training required)
3. Rule-based mask flags visibly bare areas in the image
4. AI model achieves **strictly higher IoU** than the rule baseline on the same test set
5. Comparison PNG makes it obvious why the AI is better (fewer road/river false positives)

---

## Why This Matters

If your AI model is **not better** than simple spectral rules, one of these is wrong:
- Not enough training data
- Labels are inaccurate
- Model is undertrained
- Patches don't contain enough context

The baseline gives you a **sanity check** before investing in more complex approaches.

---

## What You DON'T Build

- Machine learning (this is intentionally rule-based)
- Multi-scene compositing
- Any training loop

**Time estimate:** 2–3 hours
