# GoldMine Watch — Phase 2 Roadmap

**Objective:** Train on all 1,189 mining polygons, run inference across all of French Guiana, and serve results via a Dockerized web map with square detections.

**Key principle:** The tile cache is **unified** — tiles downloaded for training are reused for inference, and vice versa.

**Status:** One-time snapshot (no temporal monitoring).

---

## Why Square Detections?

The model outputs **per-pixel probabilities** — naturally irregular shapes. However, for a mining monitoring system, **square bounding boxes** are preferred because:

- Easier to validate area estimates
- Consistent reporting format
- Faster for experts to review
- Aligns with standard alerting grids

**Implementation:** Overlay a fixed-size grid (e.g., 128m or 256m) on the probability raster. Each grid cell becomes a detection if its mean probability exceeds a threshold. The resulting GeoJSON features are perfect squares in projected coordinates.

---

## Unified Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIFIED TILE CACHE                               │
│    data/cache/tiles/<tile_id>_<date>.tif  (used by train + infer)   │
├─────────────────────────────────────────────────────────────────────┤
│  TRAINING                            │  INFERENCE (batch, cached)   │
│  • Download tiles via cache manager  │  • Reuse same cached tiles   │
│  • Extract patches (all 1,189 mines) │  • Run predict_big_image()   │
│  • Train → models/phase2_best.pth    │  • Save predictions          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              POST-PROCESSING (Square Bounding Boxes)                 │
├─────────────────────────────────────────────────────────────────────┤
│  • Overlay a fixed-size grid on the probability raster               │
│  • For each grid cell: if mean probability > threshold → detection   │
│  • Output: Square GeoJSON features (center + side length)            │
│  • Attributes: confidence, area_m2, area_ha, detection_id            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              WEB APP (Docker + Leaflet + GeoJSON)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Base layer: OpenStreetMap                                           │
│  Overlay 1: Sentinel-2 imagery (XYZ tiles from cached COG)          │
│  Overlay 2: Mining detections (GeoJSON, red squares)                 │
│  Overlay 3: Original 1,189 labels (green, toggleable)                │
│  Popup: Confidence %, Area (ha), Detection ID                        │
│  Filter: Confidence threshold slider                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure (Target)

```
gold-watch/
├── src/goldmine_watch/
│   ├── data/
│   │   ├── tile_cache.py          # Unified tile cache manager
│   │   ├── mine_clusterer.py      # Group mines by Sentinel-2 tile
│   │   └── square_postprocess.py  # Grid-based square detections
│   └── web/
│       ├── __init__.py
│       └── server.py              # Optional FastAPI backend
├── web/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── data/
│       ├── detections.geojson     # Symlink to outputs/
│       └── labels.geojson         # Original 1,189 mines
├── data/
│   ├── cache/
│   │   ├── tiles/                 # Shared training + inference cache
│   │   └── predictions/           # Per-tile probability rasters
│   └── splits/
│       ├── train/
│       └── val/
├── models/
│   └── phase2_best.pth
├── outputs/
│   └── phase2/
│       ├── detections_french_guiana_square.geojson
│       └── qgis_project.qgz
└── design/
    └── ROADMAP_PHASE_2.md         # This file
```

---

## Milestones

### Milestone 1: Unified Tile Cache System (~2-3h)

Build the tile manager that both training and inference will use.

**Cache structure:**
```
data/cache/
├── tiles/
│   ├── S2A_20231026_T21NZF.tif
│   ├── S2A_20231026_T21NZG.tif
│   └── ...
└── predictions/
    ├── T21NZF_phase2_v1.tif
    ├── T21NZG_phase2_v1.tif
    └── ...
```

**Cache manager logic:**
```python
def get_tile(tile_id: str, date_range: str, bbox: tuple) -> Path:
    """Return cached tile or download if missing."""
    cached = find_cached_tile(tile_id, date_range)
    if cached.exists():
        return cached
    downloaded = download_scene(bbox, date_range)
    return cache_tile(downloaded, tile_id)
```

**Output:** `src/goldmine_watch/data/tile_cache.py`

---

### Milestone 2: Multi-Scene Training Dataset (~4-6h)

**Problem:** 1,189 mines span ~252×253 km. A single Sentinel-2 tile covers ~100×100 km. Need 4-6 tiles.

**Solution:**
1. **Cluster mines by Sentinel-2 UTM tile ID** (`T21NZE`, `T21NZF`, `T21NZG`, `T22NBL`, etc.)
2. **Use cache manager** to download missing tiles (check cache first)
3. **Extract patches per tile:**
   - Mine-centered: one patch per mine in that tile
   - Background: 50-100 random patches per tile
4. **Spatial train/val split** across tiles (ensure no data leakage — train and val come from different tiles)

**Output:**
- `data/splits/train/`: ~800+ patches
- `data/splits/val/`: ~200+ patches
- `data/splits/test/`: hold-out tile(s) for final evaluation

**Effort:** Mostly automated download time.

---

### Milestone 3: Train on Full Dataset (~2-3h)

Train a model that generalizes across all of French Guiana.

**Changes from Phase 1:**
- More training data (800+ patches vs. 111)
- Longer training (30-50 epochs)
- Leave-one-tile-out validation
- Consider mixed precision (MPS/AMP) for speed

**Target:** Val IoU ≥ 0.50 on held-out tile

**Output:** `models/phase2_best.pth`

---

### Milestone 4: Batch Inference with Cache (~3-4h)

Run inference on the entire territory using cached tiles.

**Steps:**
1. Build a regular grid covering French Guiana `[-54.45, 3.21, -52.18, 5.49]`
2. For each grid cell:
   - Check cache → download if missing → run inference → cache prediction
3. Skip cells with >50% cloud cover
4. Parallelize inference across CPU/GPU cores

**Output:** Per-tile predictions in `data/cache/predictions/`

---

### Milestone 5: Square Post-Processing (~1-2h)

Convert probability rasters into square bounding boxes.

**Steps:**
1. Merge per-tile predictions into a single mosaic GeoTIFF
2. Overlay a fixed-size grid (e.g., 128m or 256m cells) in projected CRS
3. For each grid cell:
   - Compute mean probability
   - If mean > threshold: emit square GeoJSON feature
4. Filter: remove cells below minimum area threshold
5. Compute attributes: confidence, area_m2, area_ha, detection_id

**Output:** `outputs/phase2/detections_french_guiana_square.geojson`

---

### Milestone 6: Web App (~2-3h)

Build a Leaflet-based web map served via Docker.

```
web/
├── Dockerfile
├── docker-compose.yml
├── index.html
├── style.css
├── app.js
└── data/
    ├── detections.geojson     # Symlink to outputs/
    └── labels.geojson         # Original 1,189 mines
```

**Features:**
- **Base layer:** OpenStreetMap (free, no API key)
- **Overlay 1:** Sentinel-2 imagery (XYZ tiles rendered from cached Cloud Optimized GeoTIFFs)
- **Overlay 2:** Mining detections (GeoJSON, displayed as red squares)
- **Overlay 3:** Original 1,189 labels (green squares, toggleable)
- **Popup:** Click a detection → see confidence score + area (ha)
- **Filter:** Confidence threshold slider (hide low-confidence detections)

**Run locally:**
```bash
cd web && python -m http.server 8000
# Open http://localhost:8000
```

---

### Milestone 7: Docker Deployment (~1-2h)

Containerize the web app for production deployment.

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY web/ /app/web/
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "web"]
```

**Build & run:**
```bash
docker build -t goldmine-watch .
docker run -p 8000:8000 -v $(pwd)/data/cache:/app/data/cache goldmine-watch
```

**Optional:** Add a lightweight FastAPI backend (`src/goldmine_watch/web/server.py`) to serve GeoJSON dynamically with filtering.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unified tile cache** | Download once, use for both training and inference. Saves bandwidth and time. |
| **Square bounding boxes** | Easier validation, consistent reporting, standard alerting format. |
| **GeoJSON (not vector tiles)** | Simpler MVP. Leaflet handles <5,000 features natively. Vector tiles can be added later. |
| **Docker deployment** | Production-ready from day one. Easy to deploy on any server. |
| **OSM base + Sentinel-2 overlay** | OSM for context, Sentinel-2 for visual verification of detections. |
| **One-time snapshot** | No temporal monitoring. Single batch inference for current state. |

---

## Estimated Timeline

| Milestone | Effort | Cumulative |
|-----------|--------|------------|
| 1. Tile cache system | 2-3h | 2-3h |
| 2. Multi-scene dataset | 4-6h | 6-9h |
| 3. Full training | 2-3h | 8-12h |
| 4. Batch inference | 3-4h | 11-16h |
| 5. Square post-processing | 1-2h | 12-18h |
| 6. Web app | 2-3h | 14-21h |
| 7. Docker deployment | 1-2h | **15-23h** |

---

## Success Criteria

- [ ] Tile cache works for both training and inference
- [ ] Model trained on patches from all 1,189 mines
- [ ] Inference covers 100% of French Guiana land area
- [ ] Detections are **square bounding boxes** (not irregular polygons)
- [ ] Web app displays OSM base + Sentinel-2 overlay + detections
- [ ] Docker container runs successfully
- [ ] Total re-download time (cached): <5 minutes
- [ ] QGIS project can be generated as alternative export

---

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| Web map stack | Leaflet (open-source, simple) |
| Base layer | OpenStreetMap + Sentinel-2 overlay |
| Tile cache | Local filesystem, unified for train + infer |
| Inference mode | Pre-computed batch (one-time snapshot) |
| Detections | Square bounding boxes |
| Web format | GeoJSON (not vector tiles) |
| Deployment | Docker container |
| Scope | Prototype / MVP |

---

*Ready to implement? Start with Milestone 1 (Tile Cache System).*
