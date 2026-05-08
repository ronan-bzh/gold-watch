# Feature 2: Cloud Masking & Quality Filtering

**Goal:** Mask out cloudy pixels using the Sentinel-2 SCL band so the model never trains on or predicts through clouds.

**Prerequisite:** Feature 1 complete (you have a real image with valid metadata).

---

## What You Build

### Source Code

`src/goldmine_watch/data/cloud_mask.py` — New module:

```python
def load_scl_band(image_path: Path) -> np.ndarray:
    """Extract the SCL (Scene Classification Layer) band from a Sentinel-2 image.
    
    Returns 2D array of SCL class codes.
    """

def create_cloud_mask(scl: np.ndarray, invalid_classes: list[int]) -> np.ndarray:
    """Create a binary mask where 1 = valid pixel, 0 = cloud/shadow/no-data.
    
    Default invalid_classes from config: [0, 3, 8, 9]
    """

def apply_cloud_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set masked pixels to 0 (or NaN) across all bands."""

def compute_valid_fraction(mask: np.ndarray) -> float:
    """Return fraction of valid (non-cloud) pixels."""
```

Update `src/goldmine_watch/data/patches.py`:
- Add `min_valid_fraction` parameter to `generate_sliding_window_patches()`
- Skip patches where cloud fraction > threshold

### Tests

`tests/unit/test_cloud_mask.py`:

```python
class TestCloudMask:
    def test_scl_to_binary_mask(self):
        """SCL classes [0,3,8,9] should become 0; others become 1."""
        
    def test_all_cloudy_returns_zero_fraction(self):
        """100% cloud cover → valid_fraction = 0.0."""
        
    def test_no_clouds_returns_one_fraction(self):
        """0% cloud cover → valid_fraction = 1.0."""
        
    def test_patch_rejected_if_too_cloudy(self):
        """Patch with 50% clouds and threshold 80% should be rejected."""
        
    def test_patch_accepted_if_clear(self):
        """Patch with 90% clear and threshold 80% should be accepted."""
```

### Functional Tests

`tests/functional/test_feature_2_cloud_masking.py`:

```python
class TestFeature2CloudMaskFlow:
    def test_load_scl_and_create_mask(self, synthetic_geotiff):
        """Load SCL from image tag and produce a correct binary mask."""

    def test_valid_fraction_all_clear(self):
        """100% clear pixels -> valid_fraction = 1.0."""

    def test_valid_fraction_all_cloudy(self):
        """100% cloudy pixels -> valid_fraction = 0.0."""

    def test_patches_filtered_by_cloud_mask(self, synthetic_geotiff, synthetic_labels):
        """Cloudy patches are rejected when cloud_mask is supplied."""

    def test_custom_invalid_classes_change_mask(self, tmp_path):
        """Custom SCL invalid classes change which pixels are masked."""

    def test_apply_cloud_mask_zeroes_invalid_pixels(self):
        """apply_cloud_mask sets invalid pixels to 0 across all bands."""
```

### Demo Script

`scripts/demo_feature2_cloud_mask.py`:

```bash
python scripts/demo_feature2_cloud_mask.py data/raw/sentinel2_scene.tif
```

Outputs three PNGs:
- `outputs/demo/cloud_rgb.png` — Original RGB
- `outputs/demo/cloud_mask.png` — Black = cloud, white = clear
- `outputs/demo/cloud_masked.png` — RGB with clouds replaced by red

Console output:
```
Image valid fraction: 78.4%
Cloudy pixels: 21.6%
```

---

## Success Criteria

1. `pytest tests/unit/test_cloud_mask.py -v` → **5 passed**
2. Demo shows a clear RGB image with clouds highlighted in red
3. The cloud mask covers >95% of visible clouds in the image (visual check)
4. A patch that is 90% cloudy is rejected; a patch that is 10% cloudy is kept

---

## What You Learn

- How cloudy your pilot area typically is
- Whether you need to download an alternative scene
- How many usable patches you can actually extract

---

## What You DON'T Build

- Multi-scene compositing (that is Feature 6)
- Model training
- Patch generation logic (just the filtering part)

**Time estimate:** 2–3 hours
