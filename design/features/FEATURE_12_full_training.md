# Feature 12: Full Territory Training

**Goal:** Train the model on patches from all 1,189 mines across French Guiana.

**Prerequisite:** Feature 11 (Multi-Scene Training Dataset) must be complete.

---

## What You Build

### Source Code

`scripts/train_phase2.py` — New script:

```python
def train_phase2(
    train_dir: str = "data/splits/train",
    val_dir: str = "data/splits/val",
    epochs: int = 50,
    batch_size: int = 8,
    output_dir: str = "models",
) -> dict:
    """Train model on full territory dataset.
    
    Uses:
    - ResNet-34 encoder
    - Class-balanced BCE loss (pos_weight)
    - ReduceLROnPlateau scheduler
    - Strong augmentation
    - Best model saved by val IoU
    """
```

### Unit Tests

`tests/unit/test_training_phase2.py`:

```python
class TestPhase2Training:
    def test_model_loads_with_correct_channels(self):
        """Model should accept 7 input channels (Sentinel-2 bands)."""
    
    def test_pos_weight_computed(self):
        """pos_weight should be > 1.0 for imbalanced data."""
    
    def test_training_reduces_loss(self):
        """Loss should decrease over first few epochs."""
    
    def test_checkpoint_saved(self, tmp_path):
        """Should save best_model.pth and epoch checkpoints."""
    
    def test_history_has_expected_keys(self):
        """History should contain train_loss, val_loss, val_iou, val_f1."""

class TestModelCapacity:
    def test_resnet34_has_more_params_than_resnet18(self):
        """ResNet-34 should be larger than ResNet-18."""
```

### Functional Tests

`tests/functional/test_feature_12_training.py`:

```python
class TestFeature12TrainingFlow:
    def test_full_training_pipeline(self):
        """End-to-end: load patches -> train 5 epochs -> save model."""
    
    def test_val_iou_increases(self):
        """Val IoU should improve over training."""
    
    def test_model_predicts_on_val_patch(self):
        """Saved model can predict on a validation patch."""
    
    def test_training_resumable(self):
        """Can resume from epoch checkpoint."""
    
    def test_overfitting_detected(self):
        """Large gap between train and val IoU indicates overfitting."""
```

### Demo Script

`scripts/demo_feature12_train.py`:

```bash
# Train on full dataset
python scripts/demo_feature12_train.py \
  --train data/splits/train \
  --val data/splits/val \
  --epochs 50 \
  --batch-size 8
```

Output:
```
Full Territory Training
=======================
Loading 1,400 training patches, 289 validation patches...
Device: mps

Training samples: 1400 | Positive pixels: 1,234,567/91,750,000 (1.35%) | pos_weight: 74.29

Epoch 01/50 — Train Loss: 0.89 | Val Loss: 0.72 | Val IoU: 0.12 | Val F1: 0.21 | LR: 0.001000
Epoch 02/50 — Train Loss: 0.67 | Val Loss: 0.58 | Val IoU: 0.24 | Val F1: 0.39 | LR: 0.001000
...
Epoch 50/50 — Train Loss: 0.12 | Val Loss: 0.18 | Val IoU: 0.68 | Val F1: 0.81 | LR: 0.000125

Training complete. Checkpoints saved to models/
Best model saved at epoch 42 (Val IoU: 0.71)
Final Val IoU: 0.68
```

---

## Configuration Updates

Update `configs/mvp.yaml`:

```yaml
model:
  architecture: "unet"
  encoder: "resnet34"
  in_channels: 7

training:
  epochs: 50
  batch_size: 8
  learning_rate: 0.001
  device: "auto"  # auto-detect cuda/mps/cpu
  
  class_balance:
    enabled: true
    max_pos_weight: 100.0
  
  scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 5
```

---

## Success Criteria

1. `pytest tests/unit/test_training_phase2.py -v` → **5 passed**
2. Model trains for 50 epochs without crash
3. Val IoU ≥ 0.50 on held-out tile
4. Best model saved automatically
5. Training completes in <3 hours on MPS/GPU
6. Can resume from checkpoint

---

## What You Learn

- Training on large, geographically diverse datasets
- Dealing with extreme class imbalance (1-3% positive pixels)
- Model capacity vs. overfitting tradeoffs

---

## What You DON'T Build

- Inference on full territory
- Web visualization
- Docker deployment

**Time estimate:** 2–3 hours (plus 1-2 hours training time)

---

## Notes

- Training time: ~1-2 hours on MPS, ~30 min on CUDA
- Monitor for overfitting: train IoU >> val IoU means model memorized patches
- If val IoU plateaus early, reduce learning rate or add more augmentation
- Save checkpoints every 5 epochs for safety
- Real-data tests require the full dataset built in Feature 11
