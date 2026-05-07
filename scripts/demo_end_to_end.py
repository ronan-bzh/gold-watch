"""End-to-end demo: synthetic data -> train -> predict -> visualize.

This script runs the full pipeline on synthetic data so you can see
results without downloading real satellite imagery.

Usage:
    python scripts/demo_end_to_end.py

Outputs:
    outputs/demo/
        labels_overlay.png       — Mining labels (red) overlaid on image
        prediction_overlay.png   — Side-by-side: original + prediction heatmap
        polygons_overlay.png     — Predicted polygons (cyan) on image
        polygons.gpkg            — Vector polygons of detected mining areas
        prediction.tif           — GeoTIFF probability raster
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.data.ingest import burn_mask
from goldmine_watch.data.patches import generate_sliding_window_patches
from goldmine_watch.inference.predict_big import predict_big_image
from goldmine_watch.inference.postprocess import postprocess
from goldmine_watch.training.train import train_patches

# Demo parameters
OUTPUT_DIR = Path("outputs/demo")
IMAGE_SIZE = 1024
NUM_BANDS = 9
PATCH_SIZE = 256
EPOCHS = 30
TARGET_CRS = "EPSG:2972"


def _percentile_stretch(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Stretch image band to 0-1 using percentile limits."""
    lo, hi = np.percentile(band, [low, high])
    stretched = (band - lo) / (hi - lo + 1e-8)
    return np.clip(stretched, 0, 1)


def _make_rgb(image: np.ndarray) -> np.ndarray:
    """Create a displayable RGB composite from multi-band image.

    Args:
        image: Array of shape (bands, height, width).

    Returns:
        RGB array of shape (height, width, 3) in 0-1 range.
    """
    rgb = np.stack(
        [_percentile_stretch(image[i]) for i in range(min(3, image.shape[0]))],
        axis=-1,
    )
    if image.shape[0] < 3:
        # Repeat bands if fewer than 3
        rgb = np.repeat(rgb, 3 // image.shape[0] + 1, axis=-1)[:, :, :3]
    return rgb


def create_synthetic_data(image_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic 1024x1024 satellite image and mining labels.

    Mining areas are given a distinct bright spectral signature so the model
    can actually learn to detect them from the background noise.

    Returns:
        Tuple of (image_array, mask_array) for visualization.
    """
    print("1. Creating synthetic satellite image and labels...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Background: low random noise
    image = np.random.randint(0, 1_500, size=(NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
    transform = from_origin(200_000, 500_000 + IMAGE_SIZE * 10, 10, 10)

    # Labels: 3 large mining polygons placed away from edges to avoid edge effects
    polygons = [
        # Large central rectangle
        Polygon([
            (200_200, 500_200), (200_500, 500_200),
            (200_500, 500_450), (200_200, 500_450),
        ]),
        # Upper-right square
        Polygon([
            (200_600, 500_600), (200_850, 500_600),
            (200_850, 500_850), (200_600, 500_850),
        ]),
        # Lower-central rectangle
        Polygon([
            (200_350, 500_500), (200_650, 500_500),
            (200_650, 500_750), (200_350, 500_750),
        ]),
    ]
    gdf = gpd.GeoDataFrame({"label": ["mining"] * 3}, geometry=polygons, crs=TARGET_CRS)
    gdf.to_file(labels_path, driver="GPKG")

    # Write initial raster so burn_mask can use it as reference
    with rasterio.open(
        image_path, "w",
        driver="GTiff", height=IMAGE_SIZE, width=IMAGE_SIZE,
        count=NUM_BANDS, dtype=image.dtype, crs=TARGET_CRS, transform=transform,
    ) as dst:
        dst.write(image)

    # Burn bright values into mining areas so the model can learn them
    mask = burn_mask(gdf, image_path).astype(bool)
    image[:, mask] = np.random.randint(8_000, 10_000, size=(NUM_BANDS, mask.sum()), dtype=np.uint16)

    with rasterio.open(image_path, "r+") as dst:
        dst.write(image)

    print(f"   Saved: {image_path}, {labels_path}")
    return image, mask


def visualize_labels(image: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    """Save a PNG of the original image with binary mask overlaid in red."""
    rgb = _make_rgb(image)

    # Overlay mask in semi-transparent red
    overlay = rgb.copy()
    overlay[mask] = [1.0, 0.2, 0.2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Synthetic Satellite Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mining Labels (Ground Truth Mask)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Image + Labels Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def visualize_prediction(
    image: np.ndarray, pred: np.ndarray, out_path: Path
) -> None:
    """Save a side-by-side of original image and prediction heatmap."""
    rgb = _make_rgb(image)

    # Threshold for overlay
    pred_mask = pred >= 0.5
    overlay = rgb.copy()
    overlay[pred_mask] = [1.0, 0.5, 0.0]  # Orange for predictions

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Model Prediction (Probability)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Image + Predictions (>0.5)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def visualize_polygons(
    image: np.ndarray, pred: np.ndarray, polygons_path: Path, out_path: Path
) -> None:
    """Save original image with predicted polygons overlaid.

    Since predicted polygons are in projected CRS, we burn them back to
    pixel space using the prediction raster as reference for alignment.
    """
    rgb = _make_rgb(image)

    # Load predicted polygons and rasterize them to pixel mask
    gdf = gpd.read_file(polygons_path)

    if gdf.empty:
        # No polygons found — just show the image with a note
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)
        ax.set_title("Predicted Mining Polygons (none found)")
        ax.axis("off")
    else:
        # Burn polygons into a pixel mask using the prediction raster
        # as a reference (same CRS, transform, and shape)
        import rasterio.features

        pred_mask = (pred >= 0.5).astype(np.uint8)
        shapes = ((geom, 1) for geom in gdf.geometry if not geom.is_empty)
        polygon_mask = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=pred.shape,
            transform=rasterio.Affine.identity(),  # pixel-to-pixel
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )

        # Create overlay: thick cyan outline for predicted polygons
        overlay = rgb.copy()
        from scipy import ndimage

        outline = np.zeros_like(polygon_mask)
        outline[1:-1, 1:-1] = (
            polygon_mask[1:-1, 1:-1]
            & ~(
                polygon_mask[0:-2, 1:-1]
                & polygon_mask[2:, 1:-1]
                & polygon_mask[1:-1, 0:-2]
                & polygon_mask[1:-1, 2:]
            )
        )
        # Dilate outline to make it thicker and more visible
        outline = ndimage.binary_dilation(outline, iterations=3)
        overlay[outline] = [0.0, 1.0, 1.0]  # Cyan outline

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Thresholded Prediction Mask")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Predicted Mining Polygons (cyan outline)")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out_path}")


def main() -> None:
    image_path = OUTPUT_DIR / "synthetic_image.tif"
    labels_path = OUTPUT_DIR / "synthetic_labels.gpkg"
    patches_dir = OUTPUT_DIR / "patches"
    model_path = OUTPUT_DIR / "model.pth"
    pred_path = OUTPUT_DIR / "prediction.tif"
    polygons_path = OUTPUT_DIR / "polygons.gpkg"

    # 1. Create synthetic data
    image, mask = create_synthetic_data(image_path, labels_path)
    visualize_labels(image, mask, OUTPUT_DIR / "labels_overlay.png")

    # 2. Generate patches
    print("\n2. Generating patches...")
    patches = generate_sliding_window_patches(
        image_path, labels_path, patch_size=PATCH_SIZE, stride=PATCH_SIZE,
        max_patches=50, output_dir=patches_dir,
    )
    print(f"   Generated {len(patches)} patches")

    # 3. Train
    print(f"\n3. Training model for {EPOCHS} epochs on CPU...")
    train_patches(patches_dir, epochs=EPOCHS, batch_size=4, lr=0.001, device="cpu")

    # Copy latest checkpoint to our output dir
    import shutil
    latest_ckpt = sorted(Path("models").glob("epoch_*.pth"))[-1]
    shutil.copy(latest_ckpt, model_path)
    print(f"   Model saved: {model_path}")

    # 4. Predict on full image
    print("\n4. Running inference on full image...")
    predict_big_image(
        image_path, model_path, pred_path,
        tile_size=PATCH_SIZE, overlap=64, in_channels=NUM_BANDS, device="cpu",
    )
    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    visualize_prediction(image, pred, OUTPUT_DIR / "prediction_overlay.png")

    # 5. Post-process to polygons
    print("\n5. Converting predictions to polygons...")
    postprocess(pred_path, polygons_path, threshold=0.5, min_area_pixels=10)
    visualize_polygons(image, pred, polygons_path, OUTPUT_DIR / "polygons_overlay.png")

    print(f"\n✅ Demo complete! All outputs saved to: {OUTPUT_DIR}/")
    print("\nYou can now open these files:")
    print(f"  - PNG images: {OUTPUT_DIR}/*.png")
    print(f"  - GeoPackage (open in QGIS): {polygons_path}")
    print(f"  - Prediction raster (open in QGIS): {pred_path}")


if __name__ == "__main__":
    main()
