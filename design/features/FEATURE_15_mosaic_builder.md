# Feature 15: Mosaic Builder

**Goal:** Merge multiple per-tile probability rasters into a single seamless mosaic covering all of French Guiana.

**Prerequisite:** Feature 13 (Batch Inference) must be complete.

---

## What You Build

### Source Code

`src/goldmine_watch/data/mosaic.py` — New module:

```python
def build_mosaic(
    raster_paths: list[Path],
    output_path: str = "outputs/phase2/mosaic.tif",
    method: str = "mean",
) -> Path:
    """Merge multiple GeoTIFFs into a single mosaic.
    
    Handles overlapping edges by averaging or taking maximum.
    """

def get_mosaic_bounds(raster_paths: list[Path]) -> tuple[float, float, float, float]:
    """Compute combined bounds of all input rasters."""

def validate_mosaic(mosaic_path: Path) -> dict:
    """Check mosaic for gaps, artifacts, or invalid values."""
```

### Unit Tests

`tests/unit/test_mosaic.py`:

```python
class TestBuildMosaic:
    def test_mosaic_created(self, tmp_path):
        """Should create a valid GeoTIFF."""
    
    def test_mosaic_covers_all_inputs(self, tmp_path):
        """Mosaic bounds should encompass all input bounds."""
    
    def test_no_gaps_in_mosaic(self, tmp_path):
        """Mosaic should have no nodata holes."""

class TestGetMosaicBounds:
    def test_computes_union_of_bounds(self):
        """Should return min of mins and max of maxes."""
```

### Functional Tests

`tests/functional/test_feature_15_mosaic.py`:

```python
class TestFeature15MosaicFlow:
    def test_full_mosaic_build(self):
        """Merge all 5 tile predictions into one mosaic."""
    
    def test_mosaic_has_correct_crs(self):
        """Mosaic CRS should match input tiles."""
    
    def test_mosaic_values_in_valid_range(self):
        """All pixel values should be [0, 1]."""
```

### Demo Script

`scripts/demo_feature15_mosaic.py`:

```bash
python scripts/demo_feature15_mosaic.py \
  --input outputs/phase2/ \
  --output outputs/phase2/mosaic.tif
```

Output:
```
Mosaic Builder
==============
Loading 5 prediction rasters...
  T21NZE: 10980x10980 px
  T21NZF: 10980x10980 px
  T21NZG: 10980x10980 px
  T22NBL: 10980x10980 px
  T22NBM: 10980x10980 px

Merging with method: mean
Output: 25000x25300 px | EPSG:32622
File size: 2.4 GB

Validation:
  No gaps detected
  Value range: [0.000, 1.000]
  Mean probability: 0.08

Saved to outputs/phase2/mosaic.tif
```

---

## Success Criteria

1. `pytest tests/unit/test_mosaic.py -v` → **3 passed**
2. Mosaic covers all of French Guiana without gaps
3. No seams or artifacts at tile boundaries
4. All pixel values in [0.0, 1.0]
5. File is a valid Cloud Optimized GeoTIFF (COG)
6. Processing completes in <30 minutes

---

## What You Learn

- Large raster mosaic techniques
- Handling tile boundaries and overlaps
- Cloud Optimized GeoTIFFs for web serving

---

## What You DON'T Build

- Polygonization
- Web tiles (XYZ)
- Visualization

**Time estimate:** 1–2 hours

---

## Notes

- Use `rasterio.merge.merge()` for proper handling of overlaps
- Consider building a COG for efficient web serving
- Memory: a 25,000×25,000 float32 raster = ~2.5 GB
- May need to process in chunks if memory is limited
- Real-data tests require predictions from Feature 13.
