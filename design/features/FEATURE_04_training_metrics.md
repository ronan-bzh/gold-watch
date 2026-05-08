# Feature 4: Training with Metrics & Spatial Validation

**Goal:** Train the model on real patches and compute honest metrics (IoU, F1) using a **spatial** train/validation split.

**Prerequisites:** Feature 3 (real patches saved to disk).

---

## What You Build

### Source Code

`src/goldmine_watch/training/metrics.py` — New module:

```python
def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for binary segmentation."""

def compute_f1(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute F1 score (harmonic mean of precision and recall)."""

def compute_precision_recall(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """Return (precision, recall)."""
```

`src/goldmine_watch/data/splits.py` — New module:

```python
def spatial_train_val_split(
    patches_dir: Path,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split patches into train/val based on spatial location.
    
    Groups patches by approximate geographic quadrant to ensure
    train and val come from different areas.
    """
```

Update `src/goldmine_watch/training/train.py`:
- Add `--val-patches` argument
- After each epoch, run validation loop
- Print train loss + val IoU/F1
- Save best model based on val IoU

### Tests

`tests/unit/test_metrics.py`:

```python
class TestMetrics:
    def test_perfect_prediction_iou_is_one(self):
        """pred == target → IoU = 1.0."""
        
    def test_all_wrong_iou_is_zero(self):
        """pred and target disjoint → IoU = 0.0."""
        
    def test_f1_balances_precision_recall(self):
        """F1 is 2 * (P*R) / (P+R)."""
        
    def test_metrics_on_random(self):
        """Random predictions should give IoU ≈ 0.01–0.05 on imbalanced data."""
```

`tests/unit/test_splits.py`:

```python
class TestSpatialSplit:
    def test_no_overlap_between_splits(self):
        """Train and val should have no patches in common."""
        
    def test_split_ratio_approximate(self):
        """val_ratio=0.2 → val set ≈ 20% of total."""
```

### Functional Tests

`tests/functional/test_feature_4_training.py`:

```python
class TestFeature4TrainingFlow:
    def test_train_on_patches_produces_checkpoints(self, synthetic_geotiff, synthetic_labels, tmp_path):
        """Training on patches saves checkpoint files."""

    def test_checkpoint_loads_and_predicts(self, synthetic_geotiff, synthetic_labels, tmp_path):
        """Saved checkpoint can be loaded and used for inference."""

    def test_training_loss_decreases(self, synthetic_geotiff, synthetic_labels, tmp_path):
        """Loss printed in epoch 2 should be <= epoch 1 (or close)."""

    def test_patches_with_cloud_mask_train_successfully(self, synthetic_geotiff, synthetic_labels, tmp_path):
        """Training succeeds when patches were generated with cloud masking."""

    def test_training_on_empty_patches_raises(self, tmp_path):
        """Empty patches directory should raise ValueError."""

    def test_different_batch_sizes_train_successfully(self, synthetic_geotiff, synthetic_labels, tmp_path):
        """Training works with batch sizes 1 and 4."""
```

### Demo Script

`scripts/demo_feature4_train.py`:

```bash
python scripts/demo_feature4_train.py \
  outputs/patches \
  --epochs 30 \
  --device cuda
```

Outputs:
- `outputs/training/loss_curve.png` — Train/val loss over epochs
- `outputs/training/iou_curve.png` — Val IoU over epochs
- `models/best_model.pth` — Checkpoint with highest val IoU

Console output:
```
Epoch 01/30 — Train Loss: 0.542 | Val IoU: 0.12 | Val F1: 0.21
Epoch 02/30 — Train Loss: 0.389 | Val IoU: 0.18 | Val F1: 0.30
...
Epoch 30/30 — Train Loss: 0.045 | Val IoU: 0.58 | Val F1: 0.73
Best model saved at epoch 27 (Val IoU: 0.61)
```

---

## Success Criteria

1. `pytest tests/unit/test_metrics.py -v` → **4 passed**
2. `pytest tests/unit/test_splits.py -v` → **2 passed**
3. Val IoU increases monotonically for first 10+ epochs
4. Best checkpoint is saved based on val IoU, not train loss
5. Final val IoU > 0.40 (minimum viable); > 0.55 (good)
6. Loss curves saved as PNG and look reasonable (no spikes, no divergence)

---

## What You Learn

- Whether your model is actually learning (not just memorizing)
- Whether you have enough data to generalize
- Whether spatial splits reveal overfitting

---

## What You DON'T Build

- Full-image inference
- Post-processing to polygons
- Multi-scene compositing

**Time estimate:** 3–4 hours
