# GoldMine Watch — Feature Roadmap

**Status:** Milestones 0–11 Complete | **Date:** 2026-05-07

This document indexes all feature specifications for moving from the synthetic demo to real-world deployment. Each feature is self-contained: it has a clear goal, source code to build, tests to verify, a demo to run, and explicit success criteria.

**Pick one feature. Build it. Run the demo. Move to the next.**

---

## Feature Index

| # | Feature | What It Does | Tests | Demo | Time |
|---|---|---------|--------------|-------|------|
| **0** | [Copernicus Data Space Download](FEATURE_00_copernicus_download.md) | Authenticate and download Sentinel-2 scenes from Copernicus STAC | 6 tests | `demo_feature0_download.py` | 2–3h |
| **1** | [Real Data Ingestion & Validation](FEATURE_01_real_data_ingestion.md) | Verify downloaded CRS/bands, load labels, check overlap | 5 tests | `demo_feature1_validate.py` | 2–3h |
| **2** | [Cloud Masking & Quality Filtering](FEATURE_02_cloud_masking.md) | Use SCL band to mask clouds; reject cloudy patches | 5 tests | `demo_feature2_cloud_mask.py` | 2–3h |
| **3** | [Real Patch Generation](FEATURE_03_patch_generation.md) | Chop real image into patches, burn real labels, save .npy | 5 tests | `demo_feature3_patches.py` | 2–3h |
| **4** | [Training with Metrics & Spatial Validation](FEATURE_04_training_metrics.md) | Train on real patches with IoU/F1 and spatial train/val split | 6 tests | `demo_feature4_train.py` | 3–4h |
| **5** | [Single-Scene Inference & Evaluation](FEATURE_05_inference_evaluate.md) | Run model on full real image, compute pixel-wise metrics | 3 tests | `demo_feature5_inference.py` | 2–3h |
| **6** | [Temporal Compositing](FEATURE_06_temporal_compositing.md) | Download multiple scenes, build cloud-free median composite | 3 tests | `demo_feature6_composite.py` | 3–4h |
| **7** | [Post-Processing, Polygonization & QGIS Export](FEATURE_07_postprocess_export.md) | Threshold predictions, extract polygons, export QGIS project | 5 tests | `demo_feature7_export.py` | 3–4h |
| **8** | [Spectral Rule-Based Baseline](FEATURE_08_baseline_rules.md) | Simple NDVI+BSI rules for comparison against AI | 4 tests | `demo_feature8_baseline.py` | 2–3h |
| **9** | [Unified Tile Cache](FEATURE_09_tile_cache.md) | Cache-first tile manager for training + inference | 6 tests | `demo_feature9_tile_cache.py` | 2–3h |
| **10** | [Mine Clusterer](FEATURE_10_mine_clusterer.md) | Group all 1,189 mines by Sentinel-2 tile | 4 tests | `demo_feature10_clusterer.py` | 1–2h |
| **11** | [Multi-Scene Training Dataset](FEATURE_11_multi_scene_dataset.md) | Download all tiles, extract patches from all mines | 5 tests | `demo_feature11_dataset.py` | 4–6h |
| **12** | [Full Territory Training](FEATURE_12_full_training.md) | Train model on all 1,189 mines | 5 tests | `demo_feature12_train.py` | 2–3h |
| **13** | [Batch Inference Engine](FEATURE_13_batch_inference.md) | Run inference on all tiles covering French Guiana | 4 tests | `demo_feature13_inference.py` | 3–4h |
| **14** | [Square Post-Processing](FEATURE_14_square_postprocess.md) | Convert heatmaps to square bounding boxes | 4 tests | `demo_feature14_square.py` | 1–2h |
| **15** | [Mosaic Builder](FEATURE_15_mosaic_builder.md) | Merge per-tile predictions into one mosaic | 3 tests | `demo_feature15_mosaic.py` | 1–2h |
| **16** | [Web Map](FEATURE_16_web_map.md) | Leaflet map with OSM, Sentinel-2, and detections | 5 tests | `demo_feature16_web.py` | 2–3h |
| **17** | [Docker Deployment](FEATURE_17_docker_deploy.md) | Containerize web app for production | 4 tests | `demo_feature17_docker.py` | 1–2h |
| **18** | [QGIS Export — Full Territory](FEATURE_18_qgis_export.md) | Export full territory as QGIS project | 3 tests | `demo_feature18_qgis.py` | 1h |

---

## Suggested Paths

### Path A: Minimal Viable Detector (Fastest)

Build Features **0 → 1 → 3 → 4 → 5 → 7**. This gives you a complete pipeline on a single scene.

```
Feature 0: Download from Copernicus
Feature 1: Validate image + labels
Feature 3: Generate patches
Feature 4: Train with metrics
Feature 5: Inference + evaluation
Feature 7: Export polygons to QGIS
```

**Time:** ~14–19 hours | **Result:** A trained model and a QGIS project you can open.

### Path B: Robust Detector (Recommended)

Build Features **0 → 1 → 2 → 3 → 4 → 5 → 8 → 7**. This adds cloud masking and a rule baseline for sanity checking.

```
Feature 0: Download from Copernicus
Feature 1: Validate image + labels
Feature 2: Cloud masking
Feature 3: Generate clean patches
Feature 4: Train with metrics
Feature 5: Inference + evaluation
Feature 8: Rule baseline (sanity check)
Feature 7: Export polygons to QGIS
```

**Time:** ~18–25 hours | **Result:** A validated detector that beats simple rules.

### Path C: Production-Grade Pipeline (Complete)

Build all features in order **0 → 1 → 2 → 6 → 3 → 4 → 5 → 8 → 7**.

```
Feature 0: Download from Copernicus
Feature 1: Validate image + labels
Feature 2: Cloud masking
Feature 6: Temporal compositing (cloud-free input)
Feature 3: Generate patches from composite
Feature 4: Train with metrics
Feature 5: Inference + evaluation
Feature 8: Rule baseline (sanity check)
Feature 7: Export polygons to QGIS
```

**Time:** ~22–31 hours | **Result:** A detector trained on clean, cloud-free composites with full evaluation.

---

## Dependency Graph

```
Feature 0 (Copernicus Download)
    │
    └──► Feature 1 (Data Validation)
            │
            ├──► Feature 2 (Cloud Masking)
            │       │
            │       ├──► Feature 3 (Patches) ──► Feature 4 (Training)
            │       │                                    │
            │       │                                    ├──► Feature 5 (Inference)
            │       │                                    │       │
            │       │                                    │       ├──► Feature 8 (Baseline)
            │       │                                    │       │
            │       │                                    │       └──► Feature 7 (Export)
            │       │
            │       └──► Feature 6 (Compositing) ────────┘
            │
            └──► Feature 8 can run independently on raw image
```

**Key insight:** Feature 6 (Temporal Compositing) is optional. You can train on a single scene if cloud cover is low.

---

## Critical Blocker

**Feature 1 requires real mining labels.** Without labeled polygons, you cannot train the model.

### How to get labels

| Source | Effort | Quality |
|--------|--------|---------|
| Local agencies (DEAL Guyane, ONF) | Medium-High | High |
| Research paper authors | Low-Medium | Variable |
| Manual digitization in QGIS | High | High — full control |

**Minimum viable:** 30–50 polygons across 2–3 scenes.

---

## Success Criteria Per Stage

| Stage | Criteria |
|-------|----------|
| After Feature 0 | Real Sentinel-2 scene downloaded, 6 bands, 10m, <20% cloud |
| After Feature 1 | Image loads, labels overlap, metadata is correct |
| After Feature 3 | ≥100 usable patches, visual check confirms alignment |
| After Feature 4 | Val IoU > 0.40, loss curve is smooth |
| After Feature 5 | Full-image inference completes, IoU > 0.40 |
| After Feature 8 | AI IoU > Rule IoU by at least +20% |
| After Feature 7 | QGIS project opens, polygons are visually plausible |

---

## Files Quick Reference

| File | Purpose |
|------|---------|
| `scripts/demo_feature0_download.py` | Download Sentinel-2 from Copernicus |
| `scripts/demo_feature1_validate.py` | Validate real image + labels |
| `scripts/demo_feature2_cloud_mask.py` | Visualize cloud masking |
| `scripts/demo_feature3_patches.py` | Show patch grid from real data |
| `scripts/demo_feature4_train.py` | Train with metrics and plots |
| `scripts/demo_feature5_inference.py` | Full-image inference + comparison |
| `scripts/demo_feature6_composite.py` | Download and composite scenes |
| `scripts/demo_feature7_export.py` | Export polygons + QGIS project |
| `scripts/demo_feature8_baseline.py` | Rule-based detection baseline |
| `configs/mvp.yaml` | Central configuration (edit before any feature) |

---

## Updating This Roadmap

As you complete features:
1. Update the status table in each feature doc
2. Note any deviations (e.g., "used 128×128 patches instead of 256×256")
3. If a feature takes significantly longer than estimated, document why

**Do not work on more than one feature at a time.** Finish one, verify it, then move to the next.
