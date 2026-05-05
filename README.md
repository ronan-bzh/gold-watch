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

3. **Run the full pipeline:**
   ```bash
   make data
   make train
   make evaluate
   make predict
   make export
   ```

4. **Run tests:**
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
