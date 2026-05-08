# Feature 3: Real Patch Generation

**Goal:** Chop the real satellite image into 256×256 patches and create matching binary masks from real labels. Save them as `.npy` files for training.

**Prerequisites:** Feature 1 (valid image + labels) and Feature 2 (cloud filtering).

---

## What You Build

### Source Code

Update `src/goldmine_watch/data/patches.py`:

```python
def generate_sliding_window_patches(
    image_path: Path,
    labels_path: Path,
    patch_size: int = 256,
    stride: int | None = None,
    max_patches: int = 500,
    output_dir: Path | None = None,
    min_valid_fraction: float = 0.8,      # NEW
    max_cloud_fraction: float = 0.2,      # NEW
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate patches with cloud filtering and valid-fraction checks.
    
    NEW: Rejects patches that are too cloudy or have too few valid pixels.
    """
```

New helper:
```python
def _compute_patch_cloud_fraction(mask_window: np.ndarray) -> float:
    """Return cloud fraction for a patch window."""
```

### Tests

`tests/unit/test_real_patches.py`:

```python
class TestRealPatchGeneration:
    def test_patches_have_correct_shape(self, synthetic_geotiff, synthetic_labels):
        """All patches should be (9, 256, 256) and (256, 256)."""
        
    def test_patches_saved_to_disk(self, tmp_path):
        """output_dir provided → .npy files exist."""
        
    def test_cloudy_patch_rejected(self):
        """Patch with cloud_fraction > max_cloud_fraction is skipped."""
        
    def test_mask_aligns_with_labels(self, synthetic_geotiff, synthetic_labels):
        """Burned mask should have 1s where labels exist."""
        
    def test_stride_controls_overlap(self):
        """stride=128 should produce ~4x more patches than stride=256."""
```

### Functional Tests

`tests/functional/test_feature_3_patch_generation.py`:

```python
class TestFeature3PatchGenerationFlow:
    def test_generate_and_save_patches(self, synthetic_geotiff, synthetic_labels, tmp_path):
        """Patches are written to disk as paired .npy files."""

    def test_patch_shapes_match_config(self, synthetic_geotiff, synthetic_labels):
        """All generated patches match the expected band and spatial dims."""

    def test_stride_doubles_patch_count(self, tmp_path):
        """stride=128 produces roughly 4x more patches than stride=256."""

    def test_stats_reported_correctly(self, synthetic_geotiff, synthetic_labels):
        """return_stats=True yields accurate generated and rejected counts."""

    def test_stats_without_rejections(self, synthetic_geotiff, synthetic_labels):
        """return_stats=True reports zero rejected when nothing is filtered."""

    def test_mask_aligns_with_labels(self, synthetic_geotiff, synthetic_labels):
        """Burned mask contains positive pixels where labels exist."""
```

### Demo Script

`scripts/demo_feature3_patches.py`:

```bash
python scripts/demo_feature3_patches.py \
  data/raw/sentinel2_scene.tif \
  data/raw/mining_surfaces.gpkg \
  --num-display 9
```

Outputs a 3×3 grid PNG:
- `outputs/demo/patch_grid.png` — 9 random patches with masks overlaid in red

Console output:
```
Generated 312 patches
Rejected 48 patches (too cloudy)
Saved to outputs/patches/
```

---

## Success Criteria

1. `pytest tests/unit/test_real_patches.py -v` → **5 passed**
2. Demo grid shows 9 diverse patches: some with mining (red overlay), some without
3. All patches are exactly 256×256 with 9 bands
4. Cloudy patches are correctly rejected
5. Mask pixels exactly match label boundaries (no misalignment)

---

## What You Learn

- How many usable training patches your scene produces
- Whether your labels are large enough to be visible in 256×256 windows
- Whether you have enough positive examples (patches containing mining)

---

## What You DON'T Build

- Model training
- Augmentation beyond horizontal flip
- Any evaluation metrics

**Time estimate:** 2–3 hours
