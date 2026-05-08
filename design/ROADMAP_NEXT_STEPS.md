# GoldMine Watch — Next Steps: From Synthetic Demo to Real-World Deployment

**Status:** Milestones 0–11 Complete | **Date:** 2026-05-07 | **Branch:** main

This document bridges the gap between the working synthetic demo and real-world deployment on French Guiana satellite imagery. It defines exactly what is needed, what is blocking, and the step-by-step path forward.

---

## 1. Current State: What Was Built and Proven

### Implemented Pipeline (Milestones 0–11)

| Milestone | Status | Proof |
|-----------|--------|-------|
| 0 | ✅ Model instantiates and saves/loads | `pytest tests/unit/test_models.py -v` |
| 1 | ✅ Config validates `patch_size>0`, `0≤cloud≤100` | `pytest tests/unit/test_data_ingest.py -v` |
| 2 | ✅ Synthetic data generation + label loading + mask burning | 7 tests passing |
| 3 | ✅ Training loop on fake data (1 image) | `python scripts/demo_end_to_end.py` |
| 4 | ✅ Label plotting script | `python scripts/plot_labels.py <labels.gpkg>` |
| 5 | ✅ STAC download of 1 Sentinel-2 scene | `python scripts/download_one_scene.py` |
| 6 | ✅ Patch extraction + visual export | `python scripts/make_patches.py <img> <labels>` |
| 7 | ✅ Sliding-window patches + augmentation + multi-epoch training | `train_patches()` with `PatchDataset` |
| 8 | ✅ Single-patch prediction + side-by-side visualization | `predict_patch()` |
| 9 | ✅ Sliding-window inference + blending on large images | `predict_big_image()` |
| 10 | ✅ Probability raster → vector polygons | `postprocess()` via `rasterio.features.shapes` |
| 11 | ✅ Polish: 24 tests pass, ruff clean, mypy clean, Makefile updated | `make test && make lint` |

### What the Synthetic Demo Proved

The end-to-end demo (`scripts/demo_end_to_end.py`) proves the following on **synthetic** data:

- ✅ U-Net model (ResNet-18, 9-band input) trains and converges (loss 0.60 → 0.012 in 30 epochs)
- ✅ Sliding-window tiling + overlap blending produces seamless predictions without edge artifacts
- ✅ Model correctly localizes high-probability regions matching ground-truth mining areas
- ✅ Post-processing thresholds predictions and extracts polygons that align with detections
- ✅ Visualization pipeline produces correctly aligned overlays (image, mask, heatmap, polygons)

### What the Synthetic Demo Did NOT Prove

These remain **untested on real data**:

- ❌ Generalization to real Sentinel-2 spectral signatures (NDVI, NDWI, BSI, atmospheric effects)
- ❌ Robustness to cloud cover, haze, seasonal vegetation changes
- ❌ Performance on varied terrain (forest canopy, rivers, roads that look like mining)
- ❌ False positive rate on non-mining areas
- ❌ Model behavior at image boundaries with partial tiles
- ❌ Training stability with class imbalance (mining is ~1-5% of pixels in real scenes)

---

## 2. Prerequisites for Real-World Application

Before running the pipeline on real French Guiana imagery, you need three things:

### 2.1 Real Satellite Image(s)

**What:** One or more Sentinel-2 L2A scenes covering your pilot area.

**How to get it:**

```bash
# Option A: Use the built-in download script
python scripts/download_one_scene.py
# → Saves to data/raw/sentinel2_scene.tif

# Option B: Manual download from Copernicus Browser
# https://browser.dataspace.copernicus.eu/
# Search for: Sentinel-2 L2A, AOI = French Guiana, max cloud = 20%
# Download bands B02, B03, B04, B08, B11, B12
```

**Configuration:**

Edit `configs/mvp.yaml`:

```yaml
geospatial:
  target_crs: "EPSG:2972"        # RGFG95 / UTM zone 22N
  pilot_bbox: [-54.1, 5.3, -53.9, 5.5]  # [min_lon, min_lat, max_lon, max_lat]
  patch_size: 256
  resolution: 10                  # 10m per pixel (Sentinel-2 native)

data:
  bands: ["B02", "B03", "B04", "B08", "B11", "B12"]
  indices: ["NDVI", "NDWI", "BSI"]
  max_cloud_cover: 20
```

### 2.2 Real Mining Labels (Ground Truth)

**This is the critical blocker.** Without labeled mining polygons, you cannot train.

**What you need:**
- A vector file (GeoPackage `.gpkg`, Shapefile `.shp`, or GeoJSON) containing polygons of known active or historical gold mining surfaces
- CRS should match the image CRS (`EPSG:2972`) or be convertible
- Each polygon should represent a single mining area (garimpo)

**How to get it:**

| Source | Effort | Quality |
|--------|--------|---------|
| **Local environmental agencies** (e.g., DEAL Guyane, ONF) | Medium-High | High — official inventories |
| **Research publications** on artisanal mining in French Guiana | Low-Medium | Variable — contact authors for data |
| **Manual digitization in QGIS** | High | High — full control, very tedious |
| **Global Forest Watch / RAISG** datasets | Low | Low-Medium — may not cover French Guiana specifically |

**Manual digitization workflow:**

1. Open the downloaded Sentinel-2 image in **QGIS**
2. Create a new GeoPackage layer with Polygon geometry, CRS = EPSG:2972
3. Zoom to suspected mining areas (look for bare soil, ponds, deforestation patterns)
4. Trace polygons around each mining site
5. Save the layer as `data/raw/mining_surfaces.gpkg`

**Minimum viable label set:**
- **Training:** ~30–50 polygons across 2–3 scenes
- **Validation:** ~10–15 polygons held out from training scenes
- **Test:** ~10–15 polygons from a completely different area (spatial holdout)

> ⚠️ **Important:** Random splitting is not enough. You must use **spatial splits** (train on one region, test on another) to avoid data leakage. Mining sites cluster geographically.

### 2.3 Compute Resources

| Task | CPU Time | GPU Time | Recommended |
|------|----------|----------|-------------|
| Download 1 scene | 2–5 min | N/A | Any machine |
| Generate patches (1024×1024) | <1 min | N/A | Any machine |
| Train 50 epochs (16 patches) | ~15 min | ~2 min | GPU strongly preferred |
| Train 50 epochs (500 patches) | ~2 hours | ~15 min | GPU required |
| Inference on 10k×10k image | ~10 min | ~1 min | GPU preferred |

**If you have a CUDA GPU:**

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Train with GPU
python -m goldmine_watch.training.train --patches <dir> --device cuda --epochs 50
```

**If you only have CPU:**

Training is possible but slow. Consider:
- Reducing image size to 512×512 for initial experiments
- Using fewer epochs (20–30) with early stopping
- Training overnight

---

## 3. Step-by-Step: Run on Real Data

### Phase A: Setup (one-time)

```bash
# 1. Install dependencies
make install          # or: pip install -e ".[dev]"

# 2. Verify everything works
make test && make lint
```

### Phase B: Acquire Data

```bash
# 1. Download satellite imagery
python scripts/download_one_scene.py
# → Check data/raw/sentinel2_scene.tif exists

# 2. Place your labels
# Copy your mining_surfaces.gpkg to:
#   data/raw/mining_surfaces.gpkg

# 3. Verify labels load correctly
python scripts/plot_labels.py data/raw/mining_surfaces.gpkg outputs/labels_check.png
# → Open outputs/labels_check.png to verify polygons look right
```

### Phase C: Preprocess

```bash
# 1. Generate patches
python scripts/make_patches.py \
  data/raw/sentinel2_scene.tif \
  data/raw/mining_surfaces.gpkg

# → Check outputs/patches/ for image_*.npy and mask_*.npy files
# → Check outputs/patches/*.png for visual verification

# 2. Inspect a few patches visually
# Open outputs/patches/patch_00_*.png to confirm:
#   - Satellite imagery looks correct (RGB colors, no black bars)
#   - White pixels overlay actual mining areas
#   - Patches cover diverse areas (mining + non-mining)
```

### Phase D: Train

```bash
# Train on patches directory
python -m goldmine_watch.training.train \
  --patches outputs/patches \
  --epochs 50 \
  --batch-size 4 \
  --lr 0.001 \
  --device cuda

# → Check models/ for epoch_*.pth checkpoints
# → Loss should decrease steadily (e.g., 0.5 → 0.05 over 50 epochs)
# → If loss plateaus early, try lowering LR or adding more data
```

### Phase E: Evaluate (before full inference)

Before running on the full image, verify the model works on a **single patch**:

```bash
# Pick one patch
python -m goldmine_watch.inference.predict \
  outputs/patches/image_0000.npy \
  models/epoch_050.pth \
  --out outputs/single_patch_pred.png

# → Open outputs/single_patch_pred.png
# → Confirm predictions align with known mining areas in that patch
```

### Phase F: Full Inference

```bash
# Run sliding-window prediction on the entire image
python -m goldmine_watch.inference.predict_big \
  data/raw/sentinel2_scene.tif \
  models/epoch_050.pth \
  --out outputs/real_prediction.tif \
  --tile-size 256 \
  --overlap 64 \
  --device cuda

# → This creates a GeoTIFF probability raster
```

### Phase G: Post-Process and Export

```bash
# Convert probability raster to vector polygons
python -m goldmine_watch.inference.postprocess \
  outputs/real_prediction.tif \
  outputs/real_polygons.gpkg \
  --threshold 0.5 \
  --min-area-pixels 50

# → Open outputs/real_polygons.gpkg in QGIS
# → Overlay on the original Sentinel-2 image
# → Visually validate: do predicted polygons match actual mining areas?
```

---

## 4. Known Limitations and Risks

### Model Limitations

| Issue | Why It Happens | Mitigation |
|-------|----------------|------------|
| **False positives on bare soil** | Mining areas and dirt roads have similar spectral signature | Add road/river masks as exclusion layers; collect more negative examples |
| **Misses small mining sites** | Model trained on 256×256 patches may miss objects < few pixels | Reduce patch size to 128 or add multi-scale training |
| **Seasonal variation** | Dry season mining ponds look different from wet season | Train on multi-temporal composites (Milestone 5 was single-scene) |
| **Cloud shadows misclassified** | Shadows have low reflectance like water in ponds | Implement SCL-based cloud masking (already in config, not wired in) |

### Data Limitations

| Issue | Impact | Solution |
|-------|--------|----------|
| **No labels** | Cannot train | See Section 2.2 — this is the primary blocker |
| **Too few labels** | Model overfits | Minimum 30–50 polygons; augment with flips/rotations |
| **Labels only in one area** | Model fails in new regions | Collect labels from 2–3 geographically distinct areas |
| **Outdated labels** | Model misses new mining | Update labels annually; consider active learning |

### Pipeline Limitations

| Issue | Impact | Solution |
|-------|--------|----------|
| **Single-scene only** | No temporal context | Implement median compositing over 3-month windows (in config, not wired) |
| **No cloud masking** | Clouds contaminate predictions | Wire SCL cloud masking from config into `data/stac.py` |
| **No validation split** | Cannot detect overfitting | Add spatial train/val split before training |
| **No metrics** | Cannot quantify performance | Add IoU, F1, precision/recall to training loop |

---

## 5. Recommended Next Actions

### Immediate (This Week)

1. **Get labels.** This is the single blocker. Contact:
   - DEAL Guyane (Direction de l'Environnement, de l'Aménagement et du Logement)
   - ONF (Office National des Forêts) — they track deforestation
   - Academic researchers who have published on artisanal mining in French Guiana

2. **Verify one real scene.** Download a scene with `scripts/download_one_scene.py` and open it in QGIS to confirm:
   - Image quality is acceptable (<20% cloud)
   - Spatial resolution is 10m
   - Your pilot area is fully covered

### Short-Term (Next 2 Weeks)

3. **Build a minimum viable training set.** Digitize or acquire 30–50 mining polygons.
4. **Run the pipeline end-to-end on real data.** Follow Phase B–G above.
5. **Validate visually in QGIS.** Check if predicted polygons match actual mining areas.
6. **Iterate on data quality.** If predictions are poor, it's almost always a data issue (not enough labels, wrong labels, or poor label quality).

### Medium-Term (Next 1–2 Months)

7. **Add validation metrics.** Implement IoU/F1 calculation and a proper spatial train/val split.
8. **Scale up data.** Acquire 2–3 more scenes with labels from different regions.
9. **Improve model architecture.** Try ResNet-34 or add attention mechanisms if underfitting.
10. **Add temporal compositing.** Implement the 3-month median composite from the config.
11. **Add exclusion layers.** Mask out roads, rivers, and urban areas to reduce false positives.

---

## 6. Success Criteria for Real-World Deployment

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Intersection over Union (IoU)** | > 0.50 | `sklearn.metrics.jaccard_score` on held-out test set |
| **F1 Score** | > 0.70 | Harmonic mean of precision and recall |
| **False Positive Rate** | < 20% | Fraction of predicted polygons that are not mining |
| **Detection Rate** | > 80% | Fraction of known mining areas detected by model |
| **Inference Time** | < 5 min per 10k×10k image | Time `predict_big_image()` on GPU |

---

## 7. Files Reference

| File | Purpose |
|------|---------|
| `scripts/demo_end_to_end.py` | **Synthetic demo** — proves pipeline works without real data |
| `scripts/download_one_scene.py` | Downloads one real Sentinel-2 scene |
| `scripts/plot_labels.py` | Visualizes label polygons on a map |
| `scripts/make_patches.py` | Generates patches from real image + labels |
| `src/goldmine_watch/training/train.py` | Training entrypoint (`--fake` or `--patches`) |
| `src/goldmine_watch/inference/predict_big.py` | Full-image sliding-window inference |
| `src/goldmine_watch/inference/postprocess.py` | Raster → vector polygon conversion |
| `configs/mvp.yaml` | Central configuration — edit before running |

---

## 8. Troubleshooting

### "No polygons found above threshold"

- Lower threshold: `--threshold 0.3`
- Check prediction raster in QGIS — are values all near 0?
- Model may be undertrained; train more epochs or verify labels align with image

### "Loss doesn't decrease"

- Verify patches contain both mining and non-mining areas
- Check that labels actually overlap with image (CRS mismatch is common)
- Try lower learning rate: `--lr 0.0001`

### "Predictions have stripe artifacts"

- Increase overlap: `--overlap 128`
- Ensure model is fully trained (loss < 0.1)
- Check for NaN values in prediction raster

### "Model runs out of memory"

- Reduce batch size: `--batch-size 2`
- Reduce tile size: `--tile-size 128`
- Use CPU if GPU memory is insufficient

---

*This document is a living guide. Update it as you iterate on real data.*
