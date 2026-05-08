# Feature 6: Temporal Compositing

**Goal:** Download multiple Sentinel-2 scenes across a 3-month window and build a cloud-free median composite.

**Prerequisites:** Feature 1 (single-scene download works).

---

## What You Build

### Source Code

Update `src/goldmine_watch/data/stac.py`:

```python
def download_composite(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    output_path: Path,
    bands: list[str] | None = None,
    max_cloud_cover: float = 20.0,
    aggregator: str = "median",
) -> Path:
    """Download multiple scenes and composite them.
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat].
        start_date: ISO date string (e.g. "2023-01-01").
        end_date: ISO date string (e.g. "2023-03-31").
        output_path: Where to save the composite GeoTIFF.
        bands: Band names to retrieve.
        max_cloud_cover: Per-scene cloud threshold.
        aggregator: "median" or "mean".
    
    Returns:
        Path to saved composite GeoTIFF.
    """
```

New helper:
```python
def _composite_scenes(stack: xr.DataArray, aggregator: str) -> xr.DataArray:
    """Compute median or mean along the time axis."""
```

### Tests

`tests/unit/test_compositing.py`:

```python
class TestCompositing:
    def test_median_reduces_clouds(self):
        """Median of cloudy+clear scenes should be closer to clear."""
        
    def test_composite_same_shape_as_input(self):
        """Output shape should match a single scene (minus time dim)."""
        
    def test_invalid_aggregator_raises(self):
        """aggregator='max' should raise ValueError."""
```

### Functional Tests

`tests/functional/test_feature_6_compositing.py`:

```python
class TestFeature6CompositingFlow:
    def test_median_composite_reduces_outliers(self, tmp_path):
        """Median of scenes with an outlier band should be closer to normal."""

    def test_composite_matches_single_scene_shape(self, tmp_path):
        """Composite of same-size scenes preserves (bands, h, w)."""

    def test_composite_written_to_geotiff(self, tmp_path):
        """Composite can be written and re-read as a valid GeoTIFF."""

    def test_mean_composite_different_from_median(self, tmp_path):
        """Mean and median composites differ for skewed data."""

    def test_composite_preserves_transform(self, tmp_path):
        """Composite GeoTIFF should have the same transform as input scenes."""
```

### Demo Script

`scripts/demo_feature6_composite.py`:

```bash
python scripts/demo_feature6_composite.py \
  --bbox "-54.1,5.3,-53.9,5.5" \
  --start 2023-01-01 \
  --end 2023-03-31 \
  --out data/raw/composite.tif
```

Outputs:
- `data/raw/composite.tif` — Median composite
- `outputs/demo/composite_comparison.png` — Side-by-side of:
  - Single scene (with clouds)
  - Median composite (cloud-free)

Console output:
```
Found 8 scenes in date range
After cloud filter: 5 scenes usable
Computing median composite...
Saved to data/raw/composite.tif
Cloud pixels reduced from 22% → 3%
```

---

## Success Criteria

1. `pytest tests/unit/test_compositing.py -v` → **3 passed**
2. Demo produces a composite with visually fewer clouds than any single scene
3. Composite has same spatial extent and resolution as input scenes
4. Download + compositing completes in < 10 minutes for 3-month window
5. Output is a valid GeoTIFF readable by `rasterio.open()`

---

## What You Learn

- How many usable scenes exist for your area in a given season
- Whether median compositing actually improves cloud coverage
- How long the download + processing pipeline takes

---

## What You DON'T Build

- Model retraining on composite (use Feature 4 with composite as input)
- Cloud shadow detection beyond SCL classes
- Harmonization between different sensors

**Time estimate:** 3–4 hours
