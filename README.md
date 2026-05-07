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

5. **Run the full pipeline:**
   ```bash
   make data
   make train
   make evaluate
   make predict
   make export
   ```

5. **Run tests:**
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

## Pipeline Stages

| Stage | Command | Description |
|---|---|---|
| Setup | `make setup` | Install dependencies and pre-commit hooks |
| Download | `python scripts/demo_feature0_download.py ...` | Download Sentinel-2 scenes from Copernicus Data Space |
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
