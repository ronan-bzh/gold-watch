# Feature 14: Square Post-Processing

**Goal:** Convert probability heatmaps into square bounding box detections on a fixed grid.

**Prerequisite:** Feature 13 (Batch Inference) must be complete.

---

## What You Build

### Source Code

`src/goldmine_watch/data/square_postprocess.py` — New module:

```python
def square_postprocess(
    probability_raster_paths: list[Path],
    grid_size_m: float = 128.0,
    threshold: float = 0.2,
    min_confidence: float = 0.3,
    output_path: str = "outputs/detections_square.geojson",
) -> Path:
    """Convert probability rasters into square bounding boxes.
    
    Steps:
    1. Merge all probability rasters into mosaic
    2. Overlay a fixed-size grid (e.g., 128m cells)
    3. For each cell: compute mean probability
    4. If mean > threshold: emit square GeoJSON feature
    5. Filter by minimum confidence
    """

def create_square_grid(
    bounds: tuple[float, float, float, float],
    grid_size_m: float,
    crs: str = "EPSG:32622",
) -> gpd.GeoDataFrame:
    """Create a regular grid of square cells."""

def compute_cell_confidence(
    raster_path: Path,
    cell_geom: Polygon,
) -> float:
    """Compute mean probability within a grid cell."""
```

### Unit Tests

`tests/unit/test_square_postprocess.py`:

```python
class TestSquarePostprocess:
    def test_outputs_geojson(self, tmp_path):
        """Should create a valid GeoJSON file."""
    
    def test_all_features_are_squares(self, tmp_path):
        """All geometries should have equal width and height."""
    
    def test_features_have_confidence_attribute(self, tmp_path):
        """Each feature should have a 'confidence' property."""
    
    def test_threshold_filters_low_confidence(self, tmp_path):
        """Features below threshold should be excluded."""

class TestCreateSquareGrid:
    def test_grid_cells_equal_size(self):
        """All cells should have the same area."""
    
    def test_grid_covers_bounds(self):
        """Grid should cover the specified bounds."""
```

### Functional Tests

`tests/functional/test_feature_14_square.py`:

```python
class TestFeature14SquareFlow:
    def test_full_postprocessing_pipeline(self):
        """Mosaic -> grid -> threshold -> GeoJSON with squares."""
    
    def test_squares_align_to_grid(self):
        """Squares should align to the defined grid."""
    
    def test_confidence_correlates_with_probability(self):
        """Higher probability -> higher confidence."""
    
    def test_different_grid_sizes_produce_different_counts(self):
        """Smaller grid -> more cells -> potentially more detections."""
```

### Demo Script

`scripts/demo_feature14_square.py`:

```bash
python scripts/demo_feature14_square.py \
  --predictions outputs/phase2/ \
  --grid-size 128 \
  --threshold 0.2
```

Output:
```
Square Post-Processing
======================
Merging 5 prediction rasters...
Mosaic: 25000x25000 px | EPSG:32622

Overlaying 128m grid...
Total cells: 48,721

Filtering by threshold (>= 0.2):
  Detections: 1,247
  Mean confidence: 0.47
  Max confidence: 0.98

Saved 1,247 square detections to outputs/detections_square.geojson
```

---

## Configuration Updates

Update `configs/mvp.yaml`:

```yaml
postprocess:
  mode: "square"  # "square" or "irregular"
  grid_size_m: 128.0
  threshold: 0.2
  min_confidence: 0.3
  output_format: "geojson"
```

---

## Success Criteria

1. `pytest tests/unit/test_square_postprocess.py -v` → **4 passed**
2. All output features are perfect squares (width == height)
3. GeoJSON has attributes: confidence, area_m2, detection_id
4. Threshold filtering works (lower threshold = more detections)
5. Grid aligns properly with UTM coordinates
6. Processing completes in <10 minutes

---

## What You Learn

- Grid-based spatial analysis
- Converting rasters to vector features
- Configurable detection granularity

---

## What You DON'T Build

- Irregular polygon extraction (already exists in Feature 7)
- Web visualization
- QGIS export

**Time estimate:** 1–2 hours

---

## Notes

- Grid size controls detection granularity:
  - 64m = very detailed, many small squares
  - 128m = balanced (recommended)
  - 256m = coarse, fewer squares
- Threshold controls sensitivity:
  - 0.1 = high recall, many false positives
  - 0.2 = balanced (recommended)
  - 0.5 = high precision, may miss small mines
- Real-data tests require predictions from Feature 13.
