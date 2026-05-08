#!/usr/bin/env python3
"""Demo script for Feature 5: Single-Scene Inference & Evaluation.

Usage:
    python scripts/demo_feature5_inference.py \
        data/raw/sentinel2_scene.tif \
        models/best_model.pth \
        data/raw/mining_surfaces.gpkg \
        --threshold 0.5
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from goldmine_watch.data.ingest import burn_mask, load_labels
from goldmine_watch.inference.evaluate import evaluate_prediction
from goldmine_watch.inference.predict_big import predict_big_image
from goldmine_watch.inference.tiler import tile_image


def create_comparison_figure(
    image_path: Path,
    pred_raster_path: Path,
    labels_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> Path:
    """Create a 3-panel comparison figure.

    Panel 1: Original RGB image
    Panel 2: Ground truth mask (green)
    Panel 3: Prediction overlay (red = predicted, yellow = overlap)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        if src.count >= 3:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        elif src.count == 2:
            bands = src.read([1, 2]).astype(np.float32)
            rgb = np.concatenate([bands, bands[1:2]], axis=0)
        elif src.count == 1:
            band = src.read(1).astype(np.float32)
            rgb = np.stack([band] * 3, axis=0)
        else:
            raise ValueError(f"Unexpected band count: {src.count}")
        height = src.height
        width = src.width

    # Normalize RGB for display
    rgb = rgb.transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    # Load ground truth mask
    gdf = load_labels(labels_path)
    mask = burn_mask(gdf, pred_raster_path).astype(bool)

    # Load prediction
    with rasterio.open(pred_raster_path) as src:
        pred = src.read(1)
    pred_binary = pred >= threshold

    # Create overlay: red = predicted, yellow = overlap, green = ground truth only
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    # Ground truth only -> green
    overlay[mask & ~pred_binary] = [0, 1, 0]
    # Prediction only -> red
    overlay[pred_binary & ~mask] = [1, 0, 0]
    # Overlap -> yellow
    overlay[mask & pred_binary] = [1, 1, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Prediction Overlay\n(Red=predicted, Yellow=overlap, Green=miss)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Parse arguments and run inference demo."""
    parser = argparse.ArgumentParser(description="Demo inference and evaluation")
    parser.add_argument("image", help="Input GeoTIFF")
    parser.add_argument("model", help="Model checkpoint")
    parser.add_argument("labels", help="Ground truth labels (GeoPackage)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Evaluation threshold")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument(
        "--output-dir",
        default="outputs/demo",
        help="Directory for outputs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "inference_prediction.tif"
    comparison_path = output_dir / "inference_comparison.png"
    metrics_path = output_dir / "inference_metrics.json"

    # Infer channels from first tile
    tiles = tile_image(args.image, tile_size=args.tile_size, overlap=args.overlap)
    if not tiles:
        raise ValueError(
            f"Image {args.image} is smaller than tile_size ({args.tile_size}) "
            "and cannot be tiled."
        )
    first_tile, _ = tiles[0]
    in_channels = first_tile.shape[0]

    # Run inference
    print(f"Running inference on {args.image}...")
    predict_big_image(
        args.image,
        args.model,
        pred_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        in_channels=in_channels,
        device=args.device,
    )
    print("Blending complete.")

    # Evaluate
    print("Evaluating against ground truth...")
    metrics = evaluate_prediction(
        pred_path,
        Path(args.labels),
        threshold=args.threshold,
    )
    print(
        f"IoU: {metrics['iou']:.2f} | F1: {metrics['f1']:.2f} | "
        f"Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f}"
    )

    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Create comparison figure
    create_comparison_figure(
        Path(args.image),
        pred_path,
        Path(args.labels),
        comparison_path,
        threshold=args.threshold,
    )
    print(f"Saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
