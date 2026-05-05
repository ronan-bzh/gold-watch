# Product Requirements Document  
## MVP — AI Detection of Illegal Gold Mining Sites in French Guiana from Satellite Imagery

---

## 1. Product Overview

### 1.1 Product Name

**GoldMine Watch MVP**

### 1.2 Purpose

The purpose of this MVP is to build an AI-assisted geospatial detection system capable of identifying **potential gold mining surfaces** in French Guiana from satellite imagery.

The MVP will focus on detecting **mining-related land disturbance**, not legally proving illegality. The “illegal” qualification will be handled later through GIS cross-analysis with mining permits, protected areas, enforcement datasets, and human expert validation.

### 1.3 Problem Statement

French Guiana is affected by illegal gold mining, which causes:

- deforestation,
- river pollution,
- habitat fragmentation,
- biodiversity loss,
- social and security risks,
- mercury contamination,
- degradation of protected natural areas.

Manual monitoring over large tropical forest areas is costly, slow, and limited by difficult terrain and cloud cover. Satellite imagery provides frequent observations, but detecting small mining sites manually remains time-consuming.

The MVP aims to demonstrate that an AI model can detect candidate mining areas automatically and produce geospatial outputs usable by analysts in GIS tools.

---

## 2. MVP Objective

### 2.1 Primary Objective

Develop a first working pipeline that takes Sentinel-2 satellite imagery as input and produces geospatial predictions of **potential mining-related surfaces** in French Guiana.

### 2.2 MVP Success Definition

The MVP is successful if it can:

1. ingest mining surface labels from an existing public geospatial source,
2. retrieve or process Sentinel-2 imagery over French Guiana,
3. generate a training dataset of image patches and binary masks,
4. train a semantic segmentation model,
5. produce prediction masks on unseen areas,
6. export predictions as GIS-compatible files,
7. allow visual validation in QGIS.

### 2.3 Non-Objective

The MVP will **not** claim to determine legal status with certainty.

The MVP will detect:

> “Potential mining or gold-mining-related surface disturbance”

not:

> “Confirmed illegal gold mining site”

---

## 3. Target Users

### 3.1 Primary Users

#### Remote Sensing Analyst

A GIS or remote sensing expert who needs to identify suspicious mining-related surfaces over large forested areas.

#### Environmental Monitoring Team

A team monitoring deforestation, protected areas, river degradation, or mining expansion.

#### NGO / Research Team

A non-governmental or academic team studying the environmental impact of gold mining in French Guiana.

### 3.2 Secondary Users

#### Enforcement Support Analyst

A user who may use detected polygons as intelligence inputs, but only after human verification and cross-checking with official sources.

#### Data Scientist

A technical user who improves the model, retrains it, evaluates performance, and manages datasets.

---

## 4. User Needs

### 4.1 Remote Sensing Analyst Needs

The analyst needs to:

- load model outputs in QGIS,
- compare predictions with Sentinel-2 imagery,
- inspect predicted polygons,
- filter detections by area and confidence,
- prioritize suspicious sites for manual review.

### 4.2 Data Scientist Needs

The data scientist needs to:

- prepare satellite imagery and labels,
- generate image patches and segmentation masks,
- train and evaluate the model,
- inspect false positives and false negatives,
- improve the dataset iteratively.

### 4.3 Environmental Monitoring Team Needs

The monitoring team needs to:

- identify areas of potential recent disturbance,
- monitor mining expansion over time,
- estimate affected surface area,
- generate maps and reports.

---

## 5. Scope

### 5.1 In Scope

The MVP includes:

- French Guiana as the geographical area of interest.
- Sentinel-2 Level-2A imagery.
- Public mining surface labels from OAM / GeoGuyane or equivalent official GIS source.
- Binary semantic segmentation.
- Detection of mining-related disturbed surfaces.
- Image patch generation.
- Training, validation, and test dataset creation.
- Model training using a U-Net-like architecture.
- Export of prediction masks as GeoTIFF.
- Export of detected polygons as GeoPackage or GeoJSON.
- Visual validation in QGIS.
- Basic evaluation metrics.
- Centralized YAML configuration file consumed by all pipeline stages.
- Standard project directory structure for data, models, and outputs.
- Makefile-based pipeline orchestration for reproducible stage execution.
- Lightweight experiment tracking via per-run YAML/JSON training manifests.
- Comprehensive automated test coverage (unit, integration, and E2E).
- Sliding-window inference with overlap and blending for large-area prediction.

### 5.2 Out of Scope

The MVP excludes:

- definitive classification of illegality,
- operational enforcement decision-making,
- real-time alerts,
- fully automated legal assessment,
- integration with drone imagery,
- use of paid commercial imagery,
- SAR imagery from Sentinel-1,
- historical change detection model,
- web application dashboard,
- user authentication,
- mobile application,
- automated tasking of satellites,
- field validation workflow.

---

## 6. Assumptions

### 6.1 Data Assumptions

- Public geospatial labels of mining surfaces are available and usable.
- Sentinel-2 imagery provides sufficient visual and spectral information for a first detection model.
- Mining-related surfaces appear as detectable disturbances compared to surrounding forest.
- Some label noise is expected because historical labels may not perfectly match selected satellite image dates.

### 6.2 Technical Assumptions

- The model will be trained on image patches, not full satellite scenes.
- A GPU is available for training.
- QGIS is available for visual inspection.
- The first version can operate offline after data preparation.
- A binary segmentation model is sufficient for the MVP.
- All pipeline parameters are managed through a single centralized YAML configuration file.
- The project follows a standard Python package structure (`src/goldmine_watch/`).

### 6.3 Product Assumptions

- Users are comfortable working with GIS files.
- Users can interpret geospatial outputs.
- Human validation remains necessary before any operational use.

---

## 7. Constraints

### 7.1 Environmental Constraints

French Guiana has persistent cloud cover, which can reduce the availability of clean optical images.

### 7.2 Data Constraints

- Labels may include both legal and illegal mining surfaces.
- Labels may represent historical disturbance rather than active mining.
- Some mining sites may be too small for reliable detection at Sentinel-2 resolution.
- Sentinel-2 spatial resolution may limit the detection of narrow trails or very small camps.

### 7.3 Technical Constraints

- Sentinel-2 10 m resolution may not capture fine-grained details.
- SWIR bands require resampling from 20 m to 10 m.
- Cloud masking and compositing are required.
- Model performance depends heavily on label quality and negative sample selection.
- Multi-scene compositing can exceed available RAM; chunked or Dask-based processing is required for scalability.
- Inference on full Sentinel-2 tiles exceeds GPU memory; a sliding-window strategy with overlap and blending is mandatory.

### 7.4 Legal and Ethical Constraints

- The system must not be used as the sole basis for accusing individuals or groups of illegal activity.
- Outputs must be framed as “potential mining-related disturbance.”
- Sensitive operational information should be handled carefully.
- Any use by authorities must involve human review and official verification.

---

## 8. Dataset Requirements

### 8.1 Label Data

The MVP requires a vector dataset containing polygons of mining-related surfaces in French Guiana.

Expected properties:

- polygon geometry,
- geospatial coordinate reference system,
- date or year attribute if available,
- source metadata,
- activity or surface classification if available,
- unique identifier if available.

### 8.2 Satellite Imagery

The MVP requires Sentinel-2 Level-2A imagery.

**Primary data source:** Microsoft Planetary Computer (STAC API).  
**Fallback data source:** Copernicus Data Space Ecosystem.

Expected bands:

- Blue,
- Green,
- Red,
- Near Infrared,
- SWIR 1,
- SWIR 2.

Derived spectral indices may include:

- NDVI,
- NDWI,
- Bare Soil Index,
- NBR or equivalent disturbance index.

**Compositing strategy:**
- Cloud masking via the Sentinel-2 Scene Classification Layer (SCL).
- Mask pixels where SCL indicates cloud, cloud shadow, or no data.
- Generate a median composite over a 3-month time window.
- Use chunked or Dask-based out-of-core processing to handle large-area composites without exceeding RAM.

### 8.3 Training Samples

The MVP dataset should include:

- positive patches containing mining surfaces,
- negative patches containing non-mining areas,
- hard negative patches containing visually similar areas.

Hard negatives may include:

- river banks,
- bare soil not related to mining,
- agricultural areas,
- urban clearings,
- roads,
- natural sand bars,
- cloud shadows,
- recently cleared non-mining land.

### 8.4 Dataset Size

Minimum viable target:

- 100 positive patches,
- 100 negative patches.

Preferred MVP target:

- 300 to 500 positive patches,
- 300 to 500 negative patches.

Recommended patch size:

- 256 x 256 pixels.

Recommended spatial resolution:

- 10 meters per pixel.

---

## 9. Data Preparation Requirements

### 9.1 Label Preparation

The system must support:

- loading vector mining surface labels,
- validating geometries,
- removing empty or invalid geometries,
- filtering labels by region or date when possible,
- reprojecting all labels to a single pipeline-wide target CRS (EPSG:2972 — RGFG95),
- rasterizing polygons into binary masks using nearest-neighbor resampling,
- graceful handling of recoverable errors: skip invalid geometries with logging, reject empty outputs with a clear error message.

### 9.2 Imagery Preparation

The system must support:

- selecting Sentinel-2 scenes over a given area of interest via STAC API (Planetary Computer primary, Copernicus fallback),
- filtering scenes by cloud cover,
- creating a cloud-reduced image composite using SCL-based masking and median aggregation over a 3-month window,
- resampling all selected bands to a common 10 m resolution using bilinear interpolation,
- reprojecting all imagery to the pipeline-wide target CRS (EPSG:2972),
- clipping imagery to training areas,
- stacking bands and indices into multi-channel images,
- processing composites via chunked or Dask-based out-of-core arrays to avoid RAM exhaustion.

### 9.3 Patch Generation

The system must support:

- generating image patches from satellite imagery,
- generating corresponding binary masks,
- preserving georeferencing metadata,
- separating positive and negative patches,
- rejecting unusable patches with excessive clouds, no data, or invalid labels.

### 9.4 Dataset Splitting

The system must produce:

- training set,
- validation set,
- test set.

Recommended split:

- 70% training,
- 15% validation,
- 15% test.

The MVP should avoid putting neighboring patches in both training and test sets whenever practical.

---

## 10. Model Requirements

### 10.1 Model Type

The MVP model must be a **semantic segmentation model**.

Recommended architecture:

- U-Net or U-Net-like model.

### 10.2 Input

The model input must be a multi-channel satellite image patch.

Recommended channels:

- Sentinel-2 spectral bands,
- vegetation index,
- water index,
- bare soil index.

### 10.3 Output

The model output must be a single-channel probability mask.

Each pixel should represent the probability of belonging to:

> potential mining-related disturbed surface.

### 10.4 Classification Type

Binary segmentation:

| Class ID | Class Name |
|---:|---|
| 0 | Non-mining / background |
| 1 | Potential mining-related surface |

### 10.5 Training Requirements

The training pipeline must support:

- binary segmentation loss,
- validation after each epoch,
- model checkpointing (best validation metric),
- threshold-based mask generation,
- recording basic metrics,
- generating a per-run training manifest (YAML or JSON) containing: model config, data hash, git commit hash, hyperparameters, and metric history,
- graceful degradation on NaN/Inf loss (early stop with logging).

### 10.6 Model Selection Criteria

The selected model should be:

- simple enough to train quickly,
- compatible with small datasets,
- suitable for segmentation,
- easy to debug visually,
- deployable on standard GPU hardware.

---

## 11. Evaluation Requirements

### 11.1 Quantitative Metrics

The MVP must report:

- Precision,
- Recall,
- F1-score,
- Intersection over Union,
- false positive rate,
- false negative rate.

### 11.2 Recommended MVP Targets

Minimum target:

| Metric | Target |
|---|---:|
| Validation IoU | > 0.35 |
| F1-score | > 0.50 |
| Recall | > 0.70 |

Precision may initially be lower if the system is designed as a candidate detection tool.

### 11.3 Qualitative Evaluation

The MVP must support visual comparison of:

- Sentinel-2 RGB image,
- derived indices,
- ground-truth label mask,
- predicted probability mask,
- predicted binary mask,
- vectorized polygons.

### 11.4 Error Analysis

The MVP must document common false positives and false negatives.

Expected false positives:

- river sand bars,
- bare river banks,
- agriculture,
- roads,
- exposed soils,
- cloud shadows,
- old cleared areas.

Expected false negatives:

- very small mining sites,
- sites under cloud cover,
- partially revegetated sites,
- sites visually similar to wet soil or river sediment,
- active sites smaller than Sentinel-2 resolution.

---

## 12. Output Requirements

### 12.1 Raster Outputs

The MVP must produce:

- probability GeoTIFF,
- binary prediction GeoTIFF.

### 12.2 Vector Outputs

The MVP must produce a GIS-compatible vector file containing detected polygons.

Required attributes:

| Attribute | Description |
|---|---|
| detection_id | Unique detection identifier |
| confidence | Average probability score |
| area_m2 | Area in square meters |
| area_ha | Area in hectares |
| source_image_date | Date or period of satellite image |
| model_version | Model version used |
| geometry | Polygon geometry |

### 12.3 QGIS Project

The MVP should provide a QGIS project or layer package containing:

- base Sentinel-2 image,
- OAM labels,
- predicted raster mask,
- predicted polygons,
- styling rules for confidence levels.

**Note:** QGIS project files must use relative file paths so the project remains portable across machines. Absolute paths will break when the project is moved.

---

## 13. User Workflow

### 13.1 Data Scientist Workflow

1. Load OAM mining surface labels.
2. Select a pilot area in French Guiana.
3. Retrieve Sentinel-2 imagery for the selected area.
4. Generate cloud-reduced image composites.
5. Generate positive and negative training patches.
6. Rasterize labels into binary masks.
7. Train the segmentation model.
8. Evaluate performance on validation and test sets.
9. Predict on an unseen area.
10. Export prediction outputs.
11. Review results in QGIS.

### 13.2 GIS Analyst Workflow

1. Open the QGIS project.
2. Inspect model predictions over satellite imagery.
3. Compare predictions with original labels.
4. Filter polygons by confidence and area.
5. Mark detections as valid, invalid, or uncertain.
6. Export reviewed detections for reporting or further analysis.

---

## 14. Functional Requirements

### FR-1 — Load Mining Labels

The system must load mining surface polygons from a GIS-compatible vector file.

### FR-2 — Clean Label Geometries

The system must validate and clean label geometries before rasterization.

### FR-3 — Select Area of Interest

The system must allow selection of a pilot area within French Guiana.

### FR-4 — Retrieve Satellite Imagery

The system must retrieve or ingest Sentinel-2 Level-2A imagery over the area of interest.

### FR-5 — Generate Composite Image

The system must generate a cloud-reduced composite image from selected Sentinel-2 observations.

### FR-6 — Compute Spectral Indices

The system must compute vegetation, water, and bare-soil-related indices.

### FR-7 — Generate Training Patches

The system must create georeferenced image patches and matching binary masks.

### FR-8 — Create Dataset Splits

The system must split the dataset into training, validation, and test sets.

### FR-9 — Train Segmentation Model

The system must train a binary semantic segmentation model.

### FR-10 — Evaluate Model

The system must calculate and report segmentation metrics.

### FR-11 — Run Inference

The system must run prediction on unseen image tiles.

### FR-12 — Export Raster Predictions

The system must export probability and binary masks as GeoTIFF files.

### FR-13 — Vectorize Predictions

The system must convert binary masks into polygon features.

### FR-14 — Export Vector Predictions

The system must export detected polygons as GeoPackage or GeoJSON.

### FR-15 — Support Visual Review

The system must support visual review of outputs in QGIS.

---

## 15. Non-Functional Requirements

### 15.1 Performance

The MVP should be able to train on a small to medium dataset within a few hours on a single GPU.

### 15.2 Usability

Outputs must be readable by standard GIS software.

### 15.3 Reproducibility

The MVP must document:

- data sources,
- preprocessing parameters,
- model configuration,
- training run information,
- model version,
- image date range.

Every training run must produce a machine-readable manifest (YAML/JSON) alongside the checkpoint, capturing the exact configuration and code version used.

### 15.4 Maintainability

The pipeline should be organized into clear stages:

- data ingestion,
- preprocessing,
- dataset generation,
- training,
- inference,
- export,
- validation.

The code must follow a standard Python package structure (`src/goldmine_watch/`) with submodules for data, models, training, inference, and export.

### 15.5 Explainability

The MVP should provide human-interpretable layers:

- RGB image,
- spectral indices,
- probability mask,
- binary mask,
- polygon outputs.

### 15.6 Safety

The system must clearly state that outputs are candidate detections and require expert validation.

### 15.7 Test Coverage

The MVP must include automated tests for every pipeline stage:

- **Unit tests:** data loading, geometry validation, CRS reprojection, patch generation, model forward pass, metric computation.
- **Integration tests:** end-to-end pipeline from labels to exported predictions on a small fixture dataset.
- **Error-path tests:** invalid geometries, 100% cloud cover, missing bands, NaN loss, empty masks.
- **Smoke tests:** model instantiation, training loop start, inference on a single tile.

Target: all new code paths covered. Regression tests are mandatory for any diff that modifies existing behavior.

---

## 16. Recommended Tools

### 16.1 GIS and Visualization

| Tool | Purpose |
|---|---|
| QGIS | Visual inspection, label correction, map creation |
| GeoPackage | Storage of vector labels and detections |
| GeoTIFF | Storage of raster imagery and prediction masks |

### 16.2 Data Processing

| Tool | Purpose |
|---|---|
| Python | Main development language |
| GeoPandas | Vector geospatial processing |
| Rasterio | Raster reading, writing, rasterization |
| Xarray / rioxarray | Multidimensional raster processing |
| Dask | Out-of-core chunked array computation for large composites |
| NumPy | Array manipulation |
| Pandas | Metadata and tabular processing |

### 16.3 Satellite Data Access

| Tool | Purpose |
|---|---|
| Microsoft Planetary Computer | Primary STAC-based access to Sentinel-2 |
| Copernicus Data Space Ecosystem | Fallback official Sentinel data access |
| pystac-client | STAC API client library |
| stackstac | Load STAC items into chunked Xarray/Dask arrays |

### 16.4 Machine Learning

| Tool | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| segmentation-models-pytorch | Ready-to-use segmentation architectures |
| Albumentations | Data augmentation |
| scikit-learn | Dataset splitting and metrics support |
| TorchGeo | Geospatial datasets, samplers, and inference utilities (evaluation candidate) |

### 16.5 Annotation

| Tool | Purpose |
|---|---|
| QGIS | Geospatial label correction |
| CVAT | Image annotation and mask correction |

### 16.6 Development Tooling

| Tool | Purpose |
|---|---|
| pytest | Test framework |
| black | Code formatting |
| ruff | Fast Python linter |
| mypy | Static type checking |
| pre-commit | Git hooks for automated formatting and linting |
| Makefile | Pipeline stage orchestration |
| OmegaConf / pydantic-settings | Centralized YAML configuration loading |

---

### 16.7 Project Directory Structure

The project must use a standard directory layout to separate code, raw data, processed artifacts, and outputs:

```
goldmine-watch/
├── src/goldmine_watch/          # Python package
│   ├── data/                    # Ingestion, preprocessing, patch generation
│   ├── models/                  # Model definitions
│   ├── training/                # Training loop and checkpointing
│   ├── inference/               # Prediction and tiling
│   ├── export/                  # Raster and vector export
│   └── config.py                # Config loader
├── data/
│   ├── raw/                     # Downloaded labels and satellite imagery
│   ├── processed/               # Cleaned labels, composites, patches
│   └── splits/                  # train / val / test manifests
├── models/                      # Trained checkpoints and training manifests
├── outputs/                     # Predictions, GeoTIFFs, vector files, QGIS projects
├── configs/                     # YAML configuration files
├── tests/                       # Unit, integration, and E2E tests
├── notebooks/                   # Exploratory analysis (optional)
├── Makefile                     # Pipeline stage targets
├── pyproject.toml               # Package metadata and dependencies
├── .pre-commit-config.yaml      # Code quality hooks
├── .gitignore                   # Excludes data/ and models/ from git
└── README.md                    # Setup and usage instructions
```

`data/`, `models/`, and `outputs/` must be excluded from version control via `.gitignore`.

### 16.8 Pipeline Orchestration

Pipeline stages must be runnable through a Makefile with explicit targets:

- `make setup` — install dependencies and pre-commit hooks
- `make data` — run data ingestion and preprocessing
- `make train` — train the segmentation model
- `make evaluate` — run evaluation on the test set
- `make predict` — run inference on a pilot area
- `make export` — export predictions to GeoTIFF and vector formats
- `make test` — run the full test suite
- `make clean` — remove processed artifacts and outputs

Each target must depend on upstream stages to prevent out-of-order execution.

### 16.9 Centralized Configuration

All tunable parameters must live in a single YAML configuration file (e.g., `configs/mvp.yaml`) consumed by every pipeline stage:

- **Geospatial:** target CRS (EPSG:2972), pilot area bounding box, patch size, resolution
- **Data:** time window, cloud cover threshold, band selection, spectral indices
- **Model:** architecture, encoder, input channels, number of classes
- **Training:** learning rate, batch size, epochs, loss function, optimizer
- **Inference:** confidence threshold, minimum detection area, sliding window overlap
- **Paths:** directory roots for data, models, and outputs

The configuration loader must validate required fields and fail fast with a descriptive error on missing or invalid values.

### 16.10 Experiment Tracking

Every training run must produce a machine-readable manifest alongside the model checkpoint:

```yaml
# models/YYYYMMDD_HHMMSS_manifest.yaml
run_id: "YYYYMMDD_HHMMSS"
git_commit: "abc1234"
data_hash: "sha256:..."
config_file: "configs/mvp.yaml"
metrics:
  best_val_iou: 0.42
  best_val_f1: 0.58
model_version: "YYYYMMDD_HHMMSS_abc1234"
```

The `model_version` field must be used as the `model_version` attribute in all exported polygon files to ensure traceability.

### 16.11 Error Handling Strategy

The pipeline must handle recoverable errors gracefully:

- **Invalid geometries:** skip with a warning log; do not crash.
- **Empty labels after filtering:** raise a clear error with instructions.
- **100% cloud cover for date range:** raise `NoValidDataError` with the queried parameters.
- **Cloudy or no-data patches:** reject during patch generation with a reason logged.
- **NaN/Inf loss during training:** halt training, log the batch and config, and exit cleanly.
- **Missing checkpoint at inference:** raise `FileNotFoundError` with the expected path.
- **GPU OOM during inference:** automatically fall back to smaller tile size with a warning.

All error messages must include actionable context (file path, config value, date range) rather than raw tracebacks.

### 16.12 Inference Strategy

Inference on areas larger than GPU memory must use a sliding-window approach:

1. Divide the target raster into overlapping tiles (e.g., 256×256 px with 64 px overlap).
2. Run the model on each tile independently.
3. Blend predictions in overlapping regions using averaging or weighted blending.
4. Reassemble the full probability mask.
5. Threshold and vectorize the blended mask.

This ensures the model can produce maps of arbitrarily large pilot areas without GPU memory constraints.

### 16.13 Package Structure

The codebase must be organized as an installable Python package under `src/goldmine_watch/`:

- `goldmine_watch.data` — label loading, imagery compositing, patch generation, dataset splitting
- `goldmine_watch.models` — U-Net definition and model factory
- `goldmine_watch.training` — training loop, validation, checkpointing, manifest generation
- `goldmine_watch.inference` — sliding-window prediction, tiling, and blending
- `goldmine_watch.export` — GeoTIFF raster export, polygonization, GeoPackage/GeoJSON export
- `goldmine_watch.config` — centralized YAML config loading and validation

Tests must mirror the package structure under `tests/unit/`, `tests/integration/`, and `tests/e2e/`.

## 17. MVP Deliverables

### 17.1 Technical Deliverables

The MVP must deliver:

- prepared label dataset,
- Sentinel-2 composite imagery for pilot area,
- generated training, validation, and test patches,
- trained segmentation model,
- model evaluation report,
- prediction GeoTIFF files,
- vectorized detection GeoPackage,
- QGIS visualization project.

### 17.2 Documentation Deliverables

The MVP must include:

- data source documentation,
- preprocessing documentation,
- model configuration summary,
- evaluation report,
- known limitations,
- user guide for QGIS review.

### 17.3 Demo Deliverables

The MVP demo should show:

1. original satellite image,
2. OAM training labels,
3. model prediction mask,
4. vectorized detections,
5. confidence-based styling,
6. examples of true positives,
7. examples of false positives,
8. examples of missed detections.

---

## 18. Milestones

### Milestone 1 — Data Acquisition

**Goal:** Obtain labels and Sentinel-2 imagery.

Deliverables:

- raw OAM label file,
- pilot area definition,
- satellite imagery access confirmed.

Estimated duration:

- 1 to 2 days.

---

### Milestone 2 — Dataset Preparation

**Goal:** Generate training patches and masks.

Deliverables:

- cleaned labels,
- image composites,
- positive patches,
- negative patches,
- train / validation / test split.

Estimated duration:

- 2 to 3 days.

---

### Milestone 3 — Baseline Model

**Goal:** Train first U-Net segmentation model.

Deliverables:

- trained model checkpoint,
- validation metrics,
- sample predictions.

Estimated duration:

- 1 to 2 days.

---

### Milestone 4 — Error Analysis and Dataset Improvement

**Goal:** Improve results through hard negatives and label cleaning.

Deliverables:

- false positive list,
- false negative list,
- improved dataset,
- second model checkpoint.

Estimated duration:

- 2 days.

---

### Milestone 5 — Pilot Inference and GIS Export

**Goal:** Run model on unseen area and export GIS outputs.

Deliverables:

- probability GeoTIFF,
- binary mask GeoTIFF,
- vector detections,
- QGIS project.

Estimated duration:

- 1 to 2 days.

---

### Milestone 6 — MVP Review

**Goal:** Produce final demo and decide whether to continue.

Deliverables:

- evaluation report,
- demo map,
- limitations summary,
- next-step recommendation.

Estimated duration:

- 1 day.

---

## 19. Timeline

### Suggested 10-Day MVP Timeline

| Day | Activity |
|---:|---|
| Day 1 | Environment setup and label acquisition |
| Day 2 | Label inspection, cleaning, and pilot area selection |
| Day 3 | Sentinel-2 imagery retrieval and compositing |
| Day 4 | Patch and mask generation |
| Day 5 | Manual review and dataset split |
| Day 6 | First U-Net training run |
| Day 7 | Evaluation and visual inspection |
| Day 8 | Dataset improvement with hard negatives |
| Day 9 | Inference on pilot area and GIS export |
| Day 10 | Demo preparation and MVP review |

---

## 20. Acceptance Criteria

### 20.1 Data Acceptance Criteria

- Mining surface labels are loaded successfully.
- Satellite imagery is aligned with labels.
- At least 200 labeled patches are generated.
- Train, validation, and test splits are created.
- Dataset can be visually inspected in QGIS.

### 20.2 Model Acceptance Criteria

- Model can be trained end-to-end.
- Model produces probability masks.
- Model produces binary masks.
- Model can run inference on unseen imagery.
- Validation metrics are reported.

### 20.3 Output Acceptance Criteria

- Prediction masks are exported as GeoTIFF.
- Predicted polygons are exported as GeoPackage or GeoJSON.
- Predicted polygons include area and confidence attributes.
- Outputs can be loaded and viewed in QGIS.

### 20.4 Product Acceptance Criteria

The MVP is accepted if it demonstrates an operational proof of concept:

> A user can select a pilot area, process Sentinel-2 imagery, run the model, and inspect candidate mining detections in QGIS.

---

## 21. Risks

### 21.1 Data Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Labels are outdated | Model learns historical sites | Filter labels by date and visually review samples |
| Labels mix legal and illegal mining | Cannot classify illegality | Frame output as mining disturbance only |
| Labels are incomplete | False negatives increase | Use hard negative/positive review and iterative labeling |
| Cloud cover reduces image quality | Poor training examples | Use multi-date composites and cloud filtering |

### 21.2 Model Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Model overfits small dataset | Poor generalization | Use augmentation and spatial split |
| Many false positives on river banks | Low precision | Add hard negatives |
| Small mining sites missed | Low recall | Use smaller patch stride and threshold tuning |
| Confusion with agriculture or roads | False positives | Add diverse negative samples |

### 21.3 Product Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Users interpret output as legal proof | Ethical/legal issue | Add clear disclaimer and validation workflow |
| QGIS workflow is too technical | Low adoption | Provide QGIS project and user guide |
| Results are not actionable | MVP value reduced | Include confidence, area, and map visualization |

---

## 22. Ethical and Legal Considerations

The MVP must be positioned as a **decision-support and monitoring tool**, not an enforcement authority.

### Required Disclaimer

All model outputs must be described as:

> “Potential mining-related disturbance requiring expert validation.”

The system must not be used as the sole basis for:

- legal accusations,
- enforcement action,
- public naming of individuals,
- punitive decisions.

Any operational use must involve:

- expert remote sensing review,
- comparison with official mining permits,
- validation against recent imagery,
- field or institutional verification where appropriate.

---

## 23. Future Enhancements

After the MVP, future versions may include:

### 23.1 Detection of Potential Illegality

Add GIS cross-analysis with:

- legal mining permits,
- protected areas,
- restricted zones,
- indigenous territories,
- environmental boundaries.

### 23.2 Change Detection

Detect newly appearing mining surfaces using:

- before / after Sentinel-2 imagery,
- time-series analysis,
- Siamese U-Net,
- ChangeFormer,
- temporal anomaly detection.

### 23.3 Cloud-Resilient Detection

Add Sentinel-1 SAR imagery to reduce cloud dependency.

### 23.4 Higher-Resolution Imagery

Use commercial or open high-resolution imagery when available.

### 23.5 Human-in-the-Loop Platform

Add a web dashboard for analysts to:

- validate detections,
- correct polygons,
- mark false positives,
- export reviewed datasets,
- retrain the model.

### 23.6 Alerting System

Create periodic monitoring with:

- scheduled imagery ingestion,
- automatic inference,
- new-site detection,
- alert prioritization.

---

## 24. Open Questions

1. Which exact OAM label layer will be used for the first MVP?
2. Which pilot area should be selected first?
3. What date range should be used for Sentinel-2 imagery?
4. What minimum detection area is useful for analysts?
5. Should priority be given to recall or precision?
6. Who will manually validate the first predictions?
7. What is the acceptable false positive rate for a monitoring workflow?
8. Are legal mining permit boundaries available for later cross-analysis?
9. Should the MVP include only active-looking sites or also historical disturbed surfaces?
10. What hardware will be available for model training?

---

## 25. Recommended MVP Configuration

| Component | Recommendation |
|---|---|
| Geography | French Guiana pilot area |
| Label source | OAM / GeoGuyane mining surfaces |
| Imagery | Sentinel-2 Level-2A |
| Resolution | 10 m |
| Patch size | 256 x 256 pixels |
| Task | Binary semantic segmentation |
| Model | U-Net |
| Output | Probability mask + binary mask + polygons |
| GIS tool | QGIS |
| Minimum dataset | 200 patches |
| Preferred dataset | 600 to 1,000 patches |
| Success priority | High recall with manageable false positives |

---

## 26. Final Summary

This MVP will demonstrate whether a lightweight AI pipeline can detect mining-related disturbed surfaces in French Guiana using public satellite imagery and existing geospatial labels.

The MVP does not attempt to prove illegal activity. Instead, it produces candidate detections that can be reviewed by experts in QGIS.

The recommended first version should focus on:

- Sentinel-2 imagery,
- OAM mining surface labels,
- binary U-Net segmentation,
- GeoTIFF and GeoPackage outputs,
- visual validation by GIS analysts.

If successful, the next product phase should add change detection, Sentinel-1 SAR, legal boundary cross-analysis, and a human-in-the-loop validation interface.