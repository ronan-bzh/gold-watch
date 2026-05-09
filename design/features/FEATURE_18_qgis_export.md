# Feature 18: QGIS Export — Full Territory

**Goal:** Export the full territory mosaic and all detections as a QGIS project for expert GIS analysis.

**Prerequisite:** Feature 15 (Mosaic Builder) and Feature 14 (Square Post-Processing) must be complete.

---

## What You Build

### Source Code

Update `src/goldmine_watch/export/qgis.py` to support full territory:

```python
def create_qgis_project_full(
    mosaic_path: Path,
    detections_path: Path,
    labels_path: Path,
    output_project_path: Path,
) -> Path:
    """Create a QGIS project for the full territory.
    
    Layers:
      - Mosaic (Sentinel-2 composite, RGB)
      - Probability heatmap (single-band, pseudocolor)
      - Detections (red squares)
      - Labels (green squares)
    """
```

### Unit Tests

`tests/unit/test_qgis_export.py`:

```python
class TestQGISExportFull:
    def test_project_file_created(self, tmp_path):
        """Should create a .qgz file."""
    
    def test_project_contains_all_layers(self, tmp_path):
        """Should have 4 layers."""
    
    def test_detections_layer_red_outline(self, tmp_path):
        """Detections should render with red outline."""
```

### Functional Tests

`tests/functional/test_feature_18_qgis.py`:

```python
class TestFeature18QGISFlow:
    def test_project_opens_in_qgis(self):
        """Double-click .qgz should open in QGIS."""
    
    def test_all_layers_visible(self):
        """All 4 layers should load without errors."""
    
    def test_zoom_to_extent_works(self):
        """Zoom to layer should show all of French Guiana."""
```

### Demo Script

```bash
python scripts/demo_feature18_qgis.py \
  --mosaic outputs/phase2/mosaic.tif \
  --detections outputs/detections_square.geojson \
  --labels data/french_guiana_mines.geojson \
  --output outputs/goldmine_watch_full.qgz
```

Output:
```
QGIS Export — Full Territory
============================
Layers:
  1. Mosaic (RGB)          — 25,000 x 25,300 px
  2. Detections (squares)  — 1,247 features, red outline
  3. Labels (squares)      — 1,189 features, green outline

Saved to outputs/goldmine_watch_full.qgz
Open in QGIS to explore.
```

---

## Success Criteria

1. `pytest tests/unit/test_qgis_export.py -v` → **3 passed**
2. .qgz file opens in QGIS without errors
3. All 4 layers visible
4. Zoom to extent shows all of French Guiana
5. Detections and labels are distinguishable (red vs. green)

---

## What You Learn

- QGIS project file format (.qgz = ZIP + XML)
- Layer styling in QGIS
- GIS workflow integration

---

## What You DON'T Build

- Real-time QGIS plugin
- Database connection

**Time estimate:** 1 hour

---

## Notes

- QGIS project uses relative paths for portability
- Include layer styles (red outline for detections, green for labels)
- Add a legend showing what each layer represents
- Real-data tests require mosaic and detections from previous features.
