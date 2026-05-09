# Feature 16: Web Map

**Goal:** Build a Leaflet-based web map that displays OpenStreetMap base layer, Sentinel-2 imagery overlay, and square mining detections.

**Prerequisite:** Feature 14 (Square Post-Processing) or Feature 15 (Mosaic Builder) must be complete.

---

## What You Build

### Source Code

`web/index.html` — Main HTML file:

```html
<!DOCTYPE html>
<html>
<head>
  <title>GoldMine Watch — French Guiana</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div id="map"></div>
  <div id="controls">
    <label>Confidence threshold: <input type="range" id="threshold" min="0" max="100" value="20"></label>
    <label><input type="checkbox" id="show-labels" checked> Show original labels</label>
    <label><input type="checkbox" id="show-sentinel" checked> Show Sentinel-2</label>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

`web/app.js` — Map logic:

```javascript
// Initialize map centered on French Guiana
const map = L.map('map').setView([4.0, -53.0], 7);

// Base layer: OpenStreetMap
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// Overlay: Sentinel-2 imagery (from COG via XYZ tiles)
// Overlay: Mining detections (GeoJSON squares)
// Overlay: Original labels (GeoJSON)

// Confidence threshold slider
// Popup: Click detection -> show confidence + area
```

`web/style.css` — Styling

### Unit Tests

N/A (frontend testing with browser tools)

### Functional Tests

`tests/functional/test_feature_16_web.py`:

```python
class TestFeature16WebMap:
    def test_map_loads_without_errors(self):
        """Open index.html in browser, check console for JS errors."""
    
    def test_detections_layer_visible(self):
        """Red squares should appear on map."""
    
    def test_threshold_slider_filters_detections(self):
        """Moving slider should hide/show detections."""
    
    def test_popup_shows_confidence(self):
        """Clicking a detection should show popup with confidence."""
    
    def test_toggle_labels_works(self):
        """Checkbox should show/hide original labels."""
```

### Demo Script

```bash
# 1. Copy data to web folder
ln -s ../../outputs/detections_square.geojson web/data/detections.geojson
ln -s ../../data/french_guiana_mines.geojson web/data/labels.geojson

# 2. Start server
cd web && python -m http.server 8000

# 3. Open browser
open http://localhost:8000
```

---

## Success Criteria

1. Map loads at `http://localhost:8000`
2. OSM base layer visible
3. Red squares for detections visible
4. Green squares for labels visible (toggleable)
5. Click detection → popup with confidence % and area
6. Threshold slider filters detections in real-time
7. No JavaScript errors in console

---

## What You Learn

- Leaflet.js mapping library
- GeoJSON rendering in web browsers
- Interactive web map design

---

## What You DON'T Build

- Backend API (static files only)
- User authentication
- Mobile app

**Time estimate:** 2–3 hours

---

## Notes

- For Sentinel-2 overlay, serve COG as XYZ tiles using `titiler` or pre-render tiles.
- Keep GeoJSON small (<10 MB) for smooth rendering.
- If GeoJSON is too large, consider clustering or vector tiles (Phase 3).
- Real-data tests require detections from Feature 14.
