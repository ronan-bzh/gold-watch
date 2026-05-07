"""Patch extraction from large satellite images and label masks."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from goldmine_watch.data.cloud_mask import compute_valid_fraction, create_cloud_mask, load_scl_band
from goldmine_watch.data.ingest import burn_mask, load_labels

logger = logging.getLogger(__name__)


def make_patch(
    image_path: str | Path,
    labels_path: str | Path,
    x: int,
    y: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a single image patch and its corresponding binary mask.

    Args:
        image_path: Path to the source image GeoTIFF.
        labels_path: Path to the vector label file.
        x: Top-left column index.
        y: Top-left row index.
        size: Patch size in pixels.

    Returns:
        Tuple of (image_patch, mask_patch) as numpy arrays.
        image_patch shape: (bands, size, size)
        mask_patch shape: (size, size)
    """
    image_path = Path(image_path)
    labels_path = Path(labels_path)

    gdf = load_labels(labels_path)

    with rasterio.open(image_path) as src:
        window = Window(x, y, size, size)
        image_patch = src.read(window=window)  # (bands, size, size)
        mask = burn_mask(gdf, image_path)  # (height, width)
        mask_patch = mask[y : y + size, x : x + size]

    return image_patch, mask_patch


def generate_sliding_window_patches(
    image_path: str | Path,
    labels_path: str | Path,
    patch_size: int = 256,
    stride: int | None = None,
    max_patches: int = 50,
    output_dir: str | Path | None = None,
    cloud_mask_path: str | Path | None = None,
    min_valid_fraction: float = 0.0,
    invalid_classes: list[int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate patches using a sliding window over the image.

    Args:
        image_path: Path to the source image GeoTIFF.
        labels_path: Path to the vector label file.
        patch_size: Size of each square patch in pixels.
        stride: Step size between patches. Defaults to patch_size (no overlap).
        max_patches: Maximum number of patches to generate.
        output_dir: If provided, save each patch as .npy files to this directory.
        cloud_mask_path: Optional path to a separate SCL GeoTIFF. If provided,
            patches with valid fraction below ``min_valid_fraction`` are skipped.
        min_valid_fraction: Minimum fraction of valid (non-cloud) pixels required
            for a patch to be kept. Defaults to 0.0 (no filtering).
        invalid_classes: SCL classes to treat as invalid. Defaults to
            [0, 3, 8, 9] when *None*.

    Returns:
        List of (image_patch, mask_patch) tuples.
    """
    image_path = Path(image_path)
    labels_path = Path(labels_path)
    if stride is None:
        stride = patch_size

    gdf = load_labels(labels_path)

    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width
        mask = burn_mask(gdf, image_path)

    # Load cloud mask if requested
    cloud_mask: np.ndarray | None = None
    if min_valid_fraction > 0.0 or cloud_mask_path is not None:
        try:
            scl = load_scl_band(image_path, scl_path=cloud_mask_path)
            cloud_mask = create_cloud_mask(scl, invalid_classes=invalid_classes)
        except FileNotFoundError:
            if cloud_mask_path is not None:
                raise
            logger.warning(
                "Cloud mask filtering requested but no SCL band found for %s. "
                "Proceeding without cloud filtering.",
                image_path,
            )

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    patch_id = 0
    rejected = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            if len(patches) >= max_patches:
                break

            image_patch, mask_patch = _extract_patch(image_path, mask, x, y, patch_size)

            # Check cloud coverage if we have a cloud mask
            if cloud_mask is not None:
                patch_cloud_mask = cloud_mask[y : y + patch_size, x : x + patch_size]
                valid_frac = compute_valid_fraction(patch_cloud_mask)
                if valid_frac < min_valid_fraction:
                    rejected += 1
                    continue

            patches.append((image_patch, mask_patch))

            if output_dir is not None:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"image_{patch_id:04d}.npy", image_patch)
                np.save(out_dir / f"mask_{patch_id:04d}.npy", mask_patch)

            patch_id += 1
        if len(patches) >= max_patches:
            break

    if rejected > 0:
        logger.info(
            "Rejected %d cloudy patches (threshold %.1f%%)", rejected, min_valid_fraction * 100
        )

    return patches


def _extract_patch(
    image_path: Path,
    mask: np.ndarray,
    x: int,
    y: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a patch from an open image and precomputed mask."""
    with rasterio.open(image_path) as src:
        window = Window(x, y, size, size)
        image_patch = src.read(window=window)
    mask_patch = mask[y : y + size, x : x + size]
    return image_patch, mask_patch


def save_patch_visual(
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    output_dir: str | Path,
    prefix: str = "patch",
) -> Path:
    """Save an image patch and overlay mask as a PNG for visual inspection.

    Args:
        image_patch: Array of shape (bands, size, size).
        mask_patch: Array of shape (size, size).
        output_dir: Directory to save PNGs.
        prefix: Filename prefix.

    Returns:
        Path to the saved PNG.
    """
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simple RGB composite from first 3 bands, normalized
    rgb = image_patch[:3].transpose(1, 2, 0).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)

    # Overlay mask in red
    overlay = rgb.copy()
    overlay[mask_patch > 0] = [255, 0, 0]

    # Side-by-side
    side_by_side = np.concatenate([rgb, overlay], axis=1)
    img = Image.fromarray(side_by_side)

    out_path = output_dir / f"{prefix}.png"
    img.save(out_path)
    return out_path
