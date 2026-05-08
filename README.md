# GoldMine Watch

AI-assisted geospatial detection of potential gold mining surfaces in French Guiana from Sentinel-2 satellite imagery.

## Quick Start

1. **Install dependencies:**
   ```bash
   make setup
   ```

2. **Configure the pipeline:**
   ```bash
   cp configs/mvp.yaml configs/experiment_01.yaml
   # Edit configs/experiment_01.yaml with your pilot area and parameters
   ```

3. **Download Sentinel-2 data from Copernicus Data Space:**
   ```bash
   export COPERNICUS_CLIENT_ID="your-client-id"
   export COPERNICUS_CLIENT_SECRET="your-client-secret"
   python scripts/demo_feature0_download.py \
     --bbox "-54.1,5.3,-53.9,5.5" \
     --date "2024-01-01/2024-01-31" \
     --output data/raw/sentinel2_scene.tif
   ```

    Preview the downloaded image:
    ```bash
    python3 -c "
    import rasterio, numpy as np, matplotlib.pyplot as plt
    with rasterio.open('data/raw/sentinel2_scene.tif') as src:
        data = src.read()
        rgb = data[[2,1,0],:,:].astype(float)  # B04, B03, B02
        for i in range(3):
            b = rgb[i]
            p2, p98 = np.percentile(b[b>0], (2,98))
            rgb[i] = np.clip((b-p2)/(p98-p2), 0, 1)
        plt.imshow(np.transpose((rgb*255).astype(np.uint8), (1,2,0)))
        plt.axis('off')
        plt.savefig('data/raw/preview.png', dpi=150, bbox_inches='tight')
        print('Saved preview to data/raw/preview.png')
    "
    ```

4. **Validate image and labels:**
   ```bash
   python scripts/demo_feature1_validate.py \
     data/raw/sentinel2_scene.tif \
     data/french_guiana_mines_2023.geojson
   ```

5. **Mask clouds using the SCL band:**
   ```bash
   python scripts/demo_feature2_cloud_mask.py data/raw/sentinel2_scene.tif
   ```

   Outputs:
   - `outputs/demo/cloud_rgb.png` — Original RGB
   - `outputs/demo/cloud_mask.png` — Black = cloud, white = clear
   - `outputs/demo/cloud_masked.png` — RGB with clouds highlighted in red

6. **Generate training patches:**
   ```bash
   python scripts/demo_feature3_patches.py \
     data/raw/sentinel2_scene.tif \
     data/french_guiana_mines_2023.geojson \
     --output-dir outputs/patches \
     --patch-size 256 \
     --stride 256
   ```

7. **Train the model on patches:**
   ```bash
   python scripts/demo_feature4_train.py \
     outputs/patches \
     --epochs 30 \
     --batch-size 4 \
     --lr 0.001 \
     --device cuda \
     --checkpoint-dir models \
     --output-dir outputs/training
   ```

   **Arguments:**
   - `patches_dir` — Directory containing `image_*.npy` and `mask_*.npy` patches
   - `--epochs` — Number of training epochs (default: 30)
   - `--batch-size` — Batch size (default: 4)
   - `--lr` — Learning rate (default: 0.001)
   - `--device` — `cpu` or `cuda` (default: cpu)
   - `--checkpoint-dir` — Where to save model checkpoints (default: `models/`)
   - `--output-dir` — Where to save metric plots (default: `outputs/training/`)

   **What happens during training:**
   - Patches are automatically split into train/val using a **spatial** split (different geographic quadrants) to prevent data leakage
   - After each epoch, the model is evaluated on the validation set
   - Console output shows train loss, validation IoU, and validation F1:
     ```
     Epoch 01/30 — Train Loss: 0.5424 | Val IoU: 0.12 | Val F1: 0.21
     Epoch 02/30 — Train Loss: 0.3891 | Val IoU: 0.18 | Val F1: 0.30
     ...
     Best model saved at epoch 27 (Val IoU: 0.61)
     ```

   **Outputs:**
   - `outputs/training/loss_curve.png` — Train/val loss over epochs
   - `outputs/training/iou_curve.png` — Validation IoU over epochs
   - `models/epoch_001.pth` ... `models/epoch_030.pth` — Per-epoch checkpoints
   - `models/best_model.pth` — Checkpoint with the highest validation IoU

   **Training from Python:**
   ```python
   from goldmine_watch.training.train import train_patches

   history = train_patches(
       patches_dir="outputs/patches",
       epochs=30,
       batch_size=4,
       device="cuda",
       output_dir="models",
   )
   print(history["val_iou"])  # List of per-epoch IoU scores
   ```

8. **Run inference on a full scene:**
   ```bash
   python scripts/demo_feature5_inference.py \
     data/raw/sentinel2_scene.tif \
     models/best_model.pth \
     data/french_guiana_mines_2023.geojson \
     --threshold 0.5 \
     --device cpu
   ```

   **Arguments:**
   - `image` — Input GeoTIFF to run inference on
   - `model` — Path to the trained checkpoint (e.g. `models/best_model.pth`)
   - `labels` — Ground-truth vector labels for evaluation
   - `--threshold` — Probability threshold for binarization (default: 0.5)
   - `--tile-size` — Sliding-window tile size in pixels (default: 256)
   - `--overlap` — Overlap between adjacent tiles (default: 64)
   - `--device` — `cpu` or `cuda` (default: cpu)
   - `--output-dir` — Where to save outputs (default: `outputs/demo/`)

   **What happens during inference:**
   - The image is split into overlapping tiles (default 256 px, 64 px overlap)
   - Each tile is run through the model to produce a probability map
   - Overlapping predictions are averaged (blended) into a full-size probability raster
   - The raster is compared pixel-wise to the ground-truth labels

   **Console output:**
   ```
   Running inference on data/raw/sentinel2_scene.tif...
   Prediction saved to outputs/demo/inference_prediction.tif
   Blending complete.
   Evaluating against ground truth...
   IoU: 0.52 | F1: 0.68 | Precision: 0.61 | Recall: 0.78
   Saved metrics to outputs/demo/inference_metrics.json
   Saved comparison to outputs/demo/inference_comparison.png
   ```

   **Outputs:**
   - `outputs/demo/inference_prediction.tif` — Full-scene probability raster (GeoTIFF, float32, 0–1)
   - `outputs/demo/inference_comparison.png` — 3-panel figure:
     - Panel 1: Original RGB image
     - Panel 2: Ground truth mask (green)
     - Panel 3: Prediction overlay (red = predicted, yellow = overlap, green = missed)
   - `outputs/demo/inference_metrics.json` — Pixel-wise metrics:
     ```json
     {
       "threshold": 0.5,
       "iou": 0.52,
       "f1": 0.68,
       "precision": 0.61,
       "recall": 0.78
     }
     ```

   **Run from Python:**
   ```python
   from goldmine_watch.inference.predict_big import predict_big_image
   from goldmine_watch.inference.evaluate import evaluate_prediction
   from pathlib import Path

   pred_path = predict_big_image(
       "data/raw/sentinel2_scene.tif",
       "models/best_model.pth",
       "outputs/prediction.tif",
       tile_size=256,
       overlap=64,
       device="cpu",
   )

   metrics = evaluate_prediction(pred_path, Path("data/raw/mining_surfaces.gpkg"))
   print(metrics)
   ```

9. **Build a temporal composite (optional):**
   ```bash
   python scripts/demo_feature6_composite.py \
     --bbox "-54.1,5.3,-53.9,5.5" \
     --start 2023-01-01 \
     --end 2023-03-31 \
     --out data/raw/composite.tif
   ```
   Outputs:
   - `data/raw/composite.tif` — Median composite
   - `outputs/demo/composite_comparison.png` — Side-by-side single scene vs composite

10. **Run the rule-based baseline (sanity check):**
    ```bash
    python scripts/demo_feature8_baseline.py \
      data/raw/sentinel2_scene.tif \
      --ndvi-threshold 0.2 \
      --bsi-threshold 0.1
    ```
    Outputs:
    - `outputs/demo/baseline_mask.png` — Binary mask from rules
    - `outputs/demo/baseline_comparison.png` — AI vs rules vs ground truth
    - `outputs/demo/baseline_polygons.gpkg` — Rule-based polygons

11. **Export polygons and QGIS project:**
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

12. **Run the full pipeline:**
    ```bash
    make data
    make train
    make evaluate
    make predict
    make export
    ```

13. **Run tests:**
    ```bash
    make test
    ```

## Project Structure

```
goldmine-watch/
├── src/goldmine_watch/          # Python package
├── data/                        # Data artifacts (gitignored)
├── models/                      # Trained checkpoints (gitignored)
├── outputs/                     # Predictions and exports (gitignored)
├── configs/                     # YAML configuration files
├── tests/                       # Automated tests
├── Makefile                     # Pipeline orchestration
└── pyproject.toml               # Package metadata
```

## Configuration

All pipeline parameters are centralized in `configs/mvp.yaml`. Copy and modify this file for each experiment. The config includes:

- Geospatial settings (CRS, bounding box, resolution)
- Data sources (STAC endpoints, bands, compositing window)
- Model architecture and training hyperparameters
- Inference parameters (sliding window, threshold)
- Export settings

## Feature Demos

Each feature has a standalone demo script. Run them in order:

| Feature | Command | Description |
|---|---|---|
| 0 — Download | `python scripts/demo_feature0_download.py ...` | Download Sentinel-2 scenes from Copernicus Data Space |
| 1 — Validate | `python scripts/demo_feature1_validate.py ...` | Validate image metadata and label overlap |
| 2 — Cloud mask | `python scripts/demo_feature2_cloud_mask.py ...` | Visualize cloud masking using the SCL band |
| 3 — Patches | `python scripts/demo_feature3_patches.py ...` | Generate training patches from image + labels |
| 4 — Train | `python scripts/demo_feature4_train.py ...` | Train model with metrics and spatial validation |
| 5 — Inference | `python scripts/demo_feature5_inference.py ...` | Run full-image inference and evaluate |
| 6 — Composite | `python scripts/demo_feature6_composite.py ...` | Build cloud-free median composite |
| 7 — Export | `python scripts/demo_feature7_export.py ...` | Extract polygons and export QGIS project |
| 8 — Baseline | `python scripts/demo_feature8_baseline.py ...` | Rule-based detection sanity check |

## Pipeline Stages

| Stage | Command | Description |
|---|---|---|
| Setup | `make setup` | Install dependencies and pre-commit hooks |
| Download | `python scripts/demo_feature0_download.py ...` | Download Sentinel-2 scenes from Copernicus Data Space |
| Validate | `python scripts/demo_feature1_validate.py ...` | Validate image metadata and label overlap |
| Cloud mask | `python scripts/demo_feature2_cloud_mask.py ...` | Visualize cloud masking using the SCL band |
| Data | `make data` | Ingest labels, retrieve imagery, generate patches |
| Train | `make train` | Train the segmentation model |
| Evaluate | `make evaluate` | Compute metrics on the test set |
| Predict | `make predict` | Run inference on a pilot area |
| Export | `make export` | Export predictions to GeoTIFF and GeoPackage |
| Test | `make test` | Run the full test suite |

## Experiment Tracking

Every training run produces a YAML manifest alongside the checkpoint:

```yaml
run_id: "20240115_120000"
git_commit: "abc1234"
data_hash: "sha256:..."
config_file: "configs/experiment_01.yaml"
metrics:
  best_val_iou: 0.42
model_version: "20240115_120000_abc1234"
```

Use the `model_version` to trace exported predictions back to their training run.

## Important Disclaimer

**This system produces candidate detections of potential mining-related disturbance. It does NOT determine legal status. All outputs require expert validation before any operational use.**

## License

MIT
