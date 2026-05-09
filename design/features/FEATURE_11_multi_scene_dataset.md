# Feature 11: Multi-Scene Training Dataset

**Goal:** Download all required Sentinel-2 tiles and extract training patches from all 1,189 mines across French Guiana.

**Prerequisite:** Feature 9 (Tile Cache) and Feature 10 (Mine Clusterer) must be complete.

---

## What You Build

### Source Code

`scripts/build_training_dataset.py` — New script:

```python
def build_training_dataset(
    mines_geojson: str = "data/french_guiana_mines.geojson",
    output_dir: str = "data/splits",
    num_background_per_tile: int = 100,
    patch_size: int = 256,
    date_range: str = "2023-06-01/2023-12-31",
) -> dict:
    """Build complete training dataset from all mines.
    
    Steps:
    1. Cluster mines by Sentinel-2 tile
    2. Download each tile (using cache)
    3. Extract mine-centered patches
    4. Extract background patches
    5. Split into train/val (spatial, by tile)
    6. Save patches as .npy files
    
    Returns statistics dict.
    """
```

### Unit Tests

`tests/unit/test_build_dataset.py`:

```python
class TestBuildTrainingDataset:
    def test_output_dirs_created(self, tmp_path):
        """Should create train/ and val/ directories."""
    
    def test_patches_are_numpy_arrays(self, tmp_path):
        """Saved patches should be valid .npy files."""
    
    def test_train_val_split_ratio(self, tmp_path):
        """Should produce ~80% train, 20% val."""
    
    def test_some_positive_patches_exist(self, tmp_path):
        """At least some patches should contain mining pixels."""

class TestMineCenteredExtraction:
    def test_mine_in_patch(self, tmp_path):
        """Mine-centered patch should contain the target mine."""
    
    def test_patch_size_correct(self, tmp_path):
        """All patches should be patch_size x patch_size."""
```

### Functional Tests

`tests/functional/test_feature_11_dataset.py`:

```python
class TestFeature11DatasetFlow:
    def test_full_dataset_build(self):
        """End-to-end: cluster -> download -> extract -> split."""
    
    def test_cache_reuse_on_second_run(self):
        """Second build should reuse cached tiles, skip downloads."""
    
    def test_all_mines_represented(self):
        """At least one patch per mine (or per cluster)."""
    
    def test_background_patches_diverse(self):
        """Background patches should come from different areas."""
    
    def test_no_data_leakage(self):
        """Train and val patches should come from different tiles."""
```

### Demo Script

`scripts/demo_feature11_dataset.py`:

```bash
# Ensure credentials are set
export $(cat .env | xargs)

# Build full dataset
python scripts/demo_feature11_dataset.py \
  --mines data/french_guiana_mines.geojson \
  --output data/splits \
  --background 100
```

Output:
```
Multi-Scene Training Dataset
=============================
Loading 1,189 mines...
Clustered into 5 tiles: T21NZE, T21NZF, T21NZG, T22NBL, T22NBM

Downloading tiles (cache-first):
  [1/5] T21NZE: Downloading... Cached.
  [2/5] T21NZF: Downloading... Cached.
  [3/5] T21NZG: Downloading... Cached.
  [4/5] T22NBL: Downloading... Cached.
  [5/5] T22NBM: Downloading... Cached.

Extracting patches:
  T21NZE: 234 mines + 100 background = 334 patches
  T21NZF: 512 mines + 100 background = 612 patches
  T21NZG: 287 mines + 100 background = 387 patches
  T22NBL: 98 mines + 100 background = 198 patches
  T22NBM: 58 mines + 100 background = 158 patches

Spatial train/val split:
  Train: 1,400 patches (80%)
  Val:   289 patches (20%)
  Tiles in train: 4
  Tiles in val: 1 (T22NBM held out)

Positive patches: 892
Negative patches: 797

Saved to data/splits/train/ and data/splits/val/
```

---

## Configuration Updates

Update `configs/mvp.yaml`:

```yaml
dataset:
  source_mines: "data/french_guiana_mines.geojson"
  output_dir: "data/splits"
  patch_size: 256
  background_per_tile: 100
  train_val_ratio: [0.8, 0.2]
  
  # Spatial split: ensure train and val come from different tiles
  spatial_split: true
```

---

## Success Criteria

1. `pytest tests/unit/test_build_dataset.py -v` → **4 passed**
2. Dataset contains patches from ALL 1,189 mines
3. Train/val split is spatial (different tiles)
4. At least 800 training patches, 200 validation patches
5. Second run reuses cache (no re-downloads)
6. Some positive patches exist (mine pixels > 0)

---

## What You Learn

- How to build large-scale geospatial training datasets
- Spatial train/val splits (prevent data leakage)
- Mine-centered vs. random patch extraction strategies

---

## What You DON'T Build

- Model training
- Inference pipeline
- Web visualization

**Time estimate:** 4–6 hours (mostly download time)

---

## Authentication Setup

1. Ensure `.env` exists:
   ```bash
   export COPERNICUS_CLIENT_ID="your-client-id"
   export COPERNICUS_CLIENT_SECRET="your-client-secret"
   ```
2. Source before running:
   ```bash
   export $(cat .env | xargs)
   ```

---

## Notes

- Total download: ~5 tiles × 350 MB = ~1.75 GB
- Cache reuse makes second runs instant
- Spatial split is critical: val tile must be completely unseen during training
- Background patches should avoid areas with mines
- Real-data tests require network access and valid credentials
