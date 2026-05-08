# Feature 1: Real Data Ingestion & Validation

**Goal:** Download a real Sentinel-2 scene and verify it loads correctly with the right CRS, bands, and resolution. Load real mining labels and confirm they spatially align with the image.

**Prerequisite:** Internet connection for STAC download.

---

## What You Build

### Source Code

`src/goldmine_watch/data/validate.py` — New module with three functions:

```python
def validate_image(image_path: Path) -> dict:
    """Check a GeoTIFF has expected bands, CRS, and resolution.
    
    Returns dict with: crs, band_count, width, height, resolution, bounds.
    Raises ValueError if any check fails.
    """

def validate_labels(labels_path: Path, expected_crs: str) -> gpd.GeoDataFrame:
    """Load labels and verify CRS matches expected.
    
    Reprojects if necessary. Raises ValueError if empty or invalid.
    """

def check_spatial_overlap(image_bounds, labels_gdf) -> bool:
    """Return True if labels intersect with image bounds.
    
    Logs a warning if overlap is <50% of label area.
    """
```

### Tests

`tests/unit/test_real_data_validation.py`:

```python
class TestValidateImage:
    def test_valid_sentinel2_passes(self, synthetic_geotiff):
        """Should return metadata dict for valid image."""
        
    def test_wrong_band_count_raises(self, tmp_path):
        """Should raise ValueError if band count != 9."""
        
    def test_missing_crs_raises(self, tmp_path):
        """Should raise ValueError if CRS is undefined."""

class TestValidateLabels:
    def test_reproject_to_target_crs(self):
        """Should reproject WGS84 labels to EPSG:2972."""
        
    def test_empty_labels_raises(self):
        """Should raise ValueError for empty GeoPackage."""

class TestSpatialOverlap:
    def test_full_overlap_returns_true(self):
        """Labels inside image bounds → True."""
        
    def test_no_overlap_returns_false(self):
        """Labels far from image → False."""
```

### Functional Tests

`tests/functional/test_feature_1_validation.py`:

```python
class TestFeature1ValidationFlow:
    def test_validate_image_then_labels(self, synthetic_geotiff, synthetic_labels):
        """Sequential validation returns correct metadata for both assets."""

    def test_labels_reprojected_to_image_crs(self, tmp_path):
        """WGS84 labels are silently reprojected to the image CRS."""

    def test_full_spatial_overlap(self, tmp_path):
        """All labels inside image bounds → has_overlap=True, fraction=1.0."""

    def test_no_spatial_overlap(self, tmp_path):
        """Labels far outside image bounds → has_overlap=False."""

    def test_missing_inputs_raise(self, tmp_path):
        """Nonexistent image or label paths raise clear errors."""
```

### Demo Script

`scripts/demo_feature1_validate.py`:

```bash
python scripts/demo_feature1_validate.py \
  data/raw/sentinel2_scene.tif \
  data/raw/mining_surfaces.gpkg
```

Output:
```
✅ Image: 10980x10980 pixels, 6 bands, EPSG:32622
✅ Labels: 42 polygons, EPSG:2972 (reprojected from EPSG:4326)
✅ Overlap: 38/42 labels (90%) intersect image bounds
⚠️  4 labels fall outside image — review coordinates
```

---

## Success Criteria

1. `pytest tests/unit/test_real_data_validation.py -v` → **5 passed**
2. Running the demo on a real downloaded scene prints clean metadata with no errors
3. Labels that are in WGS84 are silently reprojected to the image CRS
4. Mismatched CRS, missing bands, or empty labels produce **clear error messages**

---

## What You Learn

- Whether your downloaded image is actually usable (right bands, right area)
- Whether your labels overlap the image at all
- How many labels you have to work with

---

## What You DON'T Build

- Cloud masking
- Patch generation
- Model training
- Any inference

**Time estimate:** 2–3 hours
