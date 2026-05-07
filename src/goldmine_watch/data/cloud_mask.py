"""Cloud masking utilities using Sentinel-2 SCL (Scene Classification Layer) band."""

import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# Default SCL invalid classes from config: no data, cloud shadow, cloud medium, cloud high
DEFAULT_INVALID_CLASSES = [0, 3, 8, 9]


def load_scl_band(image_path: Path | str, scl_path: Path | str | None = None) -> np.ndarray:
    """Extract the SCL (Scene Classification Layer) band from a Sentinel-2 image.

    Tries, in order:
    1. An explicit ``scl_path`` if provided.
    2. A sidecar file next to the image (same stem + "_SCL.tif").
    3. A band inside the image tagged with description "SCL".
    4. The last band of the image as a fallback (with a warning).

    Args:
        image_path: Path to the source image GeoTIFF.
        scl_path: Optional explicit path to a separate SCL GeoTIFF.

    Returns:
        2D array of SCL class codes with shape (height, width).

    Raises:
        FileNotFoundError: If no SCL source can be found.
    """
    image_path = Path(image_path)

    # 1. Explicit sidecar
    if scl_path is not None:
        scl_path = Path(scl_path)
        if scl_path.exists():
            with rasterio.open(scl_path) as src:
                return src.read(1)
        raise FileNotFoundError(f"Explicit SCL file not found: {scl_path}")

    # 2. Implicit sidecar
    implicit_scl = (
        image_path.with_suffix("").with_name(image_path.stem + "_SCL").with_suffix(".tif")
    )
    if implicit_scl.exists():
        with rasterio.open(implicit_scl) as src:
            return src.read(1)

    # 3. Search for SCL-tagged band inside the image
    with rasterio.open(image_path) as src:
        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1] or ""
            tags = src.tags(i)
            if desc.upper() == "SCL" or tags.get("BAND") == "SCL":
                return src.read(i)

        # 4. Fallback: if there are more than 6 bands, assume last one is SCL
        if src.count > 6:
            logger.warning(
                "No SCL band tagged; assuming band %d is SCL. " "Pass --scl-path to be explicit.",
                src.count,
            )
            return src.read(src.count)

    raise FileNotFoundError(
        f"Could not find SCL band for {image_path}. "
        "Provide a sidecar file or ensure the image contains a tagged SCL band."
    )


def create_cloud_mask(scl: np.ndarray, invalid_classes: list[int] | None = None) -> np.ndarray:
    """Create a binary mask where 1 = valid pixel, 0 = cloud/shadow/no-data.

    Args:
        scl: 2D array of SCL class codes.
        invalid_classes: SCL classes to treat as invalid. Defaults to
            [0, 3, 8, 9] (no data, cloud shadow, cloud medium probability,
            cloud high probability).

    Returns:
        Binary mask of shape (height, width) with dtype uint8.
    """
    if invalid_classes is None:
        invalid_classes = DEFAULT_INVALID_CLASSES

    mask = np.ones_like(scl, dtype=np.uint8)
    for cls in invalid_classes:
        mask[scl == cls] = 0
    return mask


def apply_cloud_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set masked pixels to 0 across all bands.

    Args:
        image: Array of shape (bands, height, width) or (height, width).
        mask: Binary mask of shape (height, width) where 0 = invalid.

    Returns:
        Image array with invalid pixels zeroed out.
    """
    masked = image.copy()
    if masked.ndim == 3:
        masked[:, mask == 0] = 0
    elif masked.ndim == 2:
        masked[mask == 0] = 0
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {masked.shape}")
    return masked


def compute_valid_fraction(mask: np.ndarray) -> float:
    """Return fraction of valid (non-cloud) pixels.

    Args:
        mask: Binary mask where 1 = valid, 0 = invalid.

    Returns:
        Float between 0.0 and 1.0.
    """
    return float(mask.sum() / mask.size)
