# Feature 7: Post-Processing, Polygonization & QGIS Export

**Goal:** Convert the probability raster into clean vector polygons and export a ready-to-open QGIS project.

**Prerequisites:** Feature 5 (inference produces a probability raster).

---

## What You Build

### Source Code

Update `src/goldmine_watch/inference/postprocess.py`:

```python
def postprocess(
    probability_raster_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    min_area_pixels: int = 50,
    min_area_m2: float | None = None,  # NEW
) -> Path:
    """Threshold raster, extract polygons, filter by area, save GeoPackage.
    
    NEW: min_area_m2 filters by real-world area using pixel resolution.
    """
```

`src/goldmine_watch/export/qgis.py` — New module:

```python
def create_qgis_project(
    image_path: Path,
    prediction_path: Path,
    polygons_path: Path,
    output_project_path: Path,
) -> Path:
    """Create a .qgz QGIS project file with all layers pre-styled.
    
    Layers:
      - Satellite image (RGB rendered)
      - Probability heatmap (pseudo-color)
      - Detected polygons (outlined, labeled with confidence)
    """
```

`src/goldmine_watch/export/csv.py` — New module:

```python
def export_polygon_metrics(polygons_path: Path, output_csv: Path) -> Path:
    """Export a CSV with columns: detection_id, area_m2, area_ha, confidence."""
```

### Tests

`tests/unit/test_postprocess.py`:

```python
class TestPostprocess:
    def test_threshold_creates_binary_mask(self):
        """Values >= 0.5 become 1, others 0."""
        
    def test_small_polygons_filtered(self):
        """Polygon < min_area_pixels should be removed."""
        
    def test_output_has_expected_columns(self):
        """GeoPackage should have geometry + area columns."""
```

`tests/unit/test_export.py`:

```python
class TestExport:
    def test_csv_has_all_columns(self):
        """CSV should have detection_id, area_m2, area_ha, confidence."""
        
    def test_qgis_project_file_created(self):
        """.qgz file should exist after export."""
```

### Functional Tests

`tests/functional/test_feature_7_postprocess.py`:

```python
class TestFeature7PostprocessFlow:
    def test_postprocess_creates_geopackage(self, tmp_path):
        """Thresholding a probability raster produces a GeoPackage."""

    def test_high_threshold_removes_polygons(self, tmp_path):
        """threshold=0.9 should remove the 0.8 square."""

    def test_min_area_filter_removes_small_polygons(self, tmp_path):
        """min_area_pixels=1000 should remove the 100x100 square."""

    def test_output_crs_matches_input(self, tmp_path):
        """Output GeoPackage should have the same CRS as the input raster."""

    def test_export_to_csv_from_geopackage(self, tmp_path):
        """Polygon attributes can be exported to CSV."""

    def test_empty_raster_produces_empty_geopackage(self, tmp_path):
        """All-zero probability raster should produce empty but valid GeoPackage."""
```

### Demo Script

`scripts/demo_feature7_export.py`:

```bash
python scripts/demo_feature7_export.py \
  data/raw/sentinel2_scene.tif \
  outputs/real_prediction.tif \
  models/best_model.pth \
  --threshold 0.5 \
  --min-area-m2 500
```

Outputs:
- `outputs/real_polygons.gpkg` — Vector polygons with attributes
- `outputs/real_polygons.csv` — Tabular export
- `outputs/detection_project.qgz` — QGIS project (open with `qgis outputs/detection_project.qgz`)

Console output:
```
Thresholding at 0.5...
Extracted 127 raw polygons
Filtered to 89 polygons (area >= 500 m²)
Saved GeoPackage: outputs/real_polygons.gpkg
Saved CSV: outputs/real_polygons.csv
Created QGIS project: outputs/detection_project.qgz
Total detected area: 342.7 ha
```

---

## Success Criteria

1. `pytest tests/unit/test_postprocess.py -v` → **3 passed**
2. `pytest tests/unit/test_export.py -v` → **2 passed**
3. Opening `detection_project.qgz` in QGIS shows:
   - Satellite image as base layer
   - Probability heatmap as overlay
   - Polygons as outlined vectors
4. CSV contains sensible area values (not NaN or negative)
5. Filtering by `min_area_m2` removes small noise polygons

---

## What You Learn

- How many detections the model produces at your chosen threshold
- How much total area is flagged as potential mining
- Whether the QGIS project is actually useful for human review

---

## What You DON'T Build

- Change detection (compare two time periods)
- Confidence calibration
- Automated report generation

**Time estimate:** 3–4 hours
