#!/usr/bin/env python3
"""Demo script for Feature 8: Spectral Rule-Based Baseline.

Usage:
    python scripts/demo_feature8_baseline.py \
        data/raw/sentinel2_scene.tif \
        --labels data/raw/mining_surfaces.gpkg \
        --model models/best_model.pth \
        --output-dir outputs/demo
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml

from goldmine_watch.baseline.rules import detect_mining_rules, rules_to_polygons
from goldmine_watch.data.ingest import burn_mask, load_labels
from goldmine_watch.inference.evaluate import evaluate_prediction
from goldmine_watch.inference.postprocess import postprocess
from goldmine_watch.inference.predict_big import predict_big_image
from goldmine_watch.inference.tiler import tile_image
from goldmine_watch.training.metrics import compute_iou


def _load_rgb(image_path: Path) -> np.ndarray:
    """Load RGB channels from a multiband GeoTIFF for display."""
    with rasterio.open(image_path) as src:
        band_names = [src.descriptions[i] or "" for i in range(src.count)]
        # Try to read true-color RGB (B04=Red, B03=Green, B02=Blue)
        if "B04" in band_names and "B03" in band_names and "B02" in band_names:
            red_idx = band_names.index("B04") + 1
            green_idx = band_names.index("B03") + 1
            blue_idx = band_names.index("B02") + 1
            rgb = src.read([red_idx, green_idx, blue_idx]).astype(np.float32)
        elif src.count >= 3:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        elif src.count == 2:
            bands = src.read([1, 2]).astype(np.float32)
            rgb = np.concatenate([bands, bands[1:2]], axis=0)
        else:
            band = src.read(1).astype(np.float32)
            rgb = np.stack([band] * 3, axis=0)

    rgb = rgb.transpose(1, 2, 0)
    # Normalize reflectance values (typical Sentinel-2 range 0-10000)
    rgb = np.clip(rgb / 3000.0, 0, 1)
    return np.asarray(rgb, dtype=np.float32)


def create_comparison_figure(
    image_path: Path,
    pred_raster_path: Path,
    rule_mask: np.ndarray,
    labels_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> Path:
    """Create a 4-panel comparison figure.

    Panels:
        1. Original RGB image
        2. AI model prediction (binary)
        3. Rule-based prediction (binary)
        4. Ground truth mask (binary)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb(image_path)
    height, width = rgb.shape[:2]

    # Load AI prediction
    with rasterio.open(pred_raster_path) as src:
        ai_pred = src.read(1)
    ai_binary = ai_pred >= threshold

    # Load ground truth mask
    gdf = load_labels(labels_path)
    gt_mask = burn_mask(gdf, pred_raster_path).astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ai_binary, cmap="Reds", vmin=0, vmax=1)
    axes[0, 1].set_title("AI Model Prediction")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rule_mask, cmap="Oranges", vmin=0, vmax=1)
    axes[1, 0].set_title("Rule-Based Prediction")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_mask, cmap="Greens", vmin=0, vmax=1)
    axes[1, 1].set_title("Ground Truth")
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Parse arguments and run the baseline demo."""
    parser = argparse.ArgumentParser(
        description="Demo: Spectral rule-based baseline vs AI model"
    )
    parser.add_argument("image", help="Input GeoTIFF")
    parser.add_argument("--labels", required=True, help="Ground truth labels (GeoPackage)")
    parser.add_argument("--model", required=True, help="Model checkpoint (.pth)")
    parser.add_argument(
        "--config", default="configs/mvp.yaml", help="Pipeline configuration YAML"
    )
    parser.add_argument(
        "--output-dir", default="outputs/demo", help="Directory for outputs"
    )
    parser.add_argument("--device", default="cpu", help="Device for AI inference")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    baseline_cfg = cfg.get("baseline", {})
    ndvi_threshold = baseline_cfg.get("ndvi_threshold", 0.2)
    bsi_threshold = baseline_cfg.get("bsi_threshold", 0.1)

    inference_cfg = cfg.get("inference", {})
    tile_size = inference_cfg.get("tile_size", 256)
    overlap = inference_cfg.get("overlap", 64)
    threshold = inference_cfg.get("threshold", 0.5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / "baseline_mask.png"
    comparison_path = output_dir / "baseline_comparison.png"
    polygons_path = output_dir / "baseline_polygons.gpkg"
    ai_pred_path = output_dir / "ai_prediction.tif"
    ai_polygons_path = output_dir / "ai_polygons.gpkg"

    # 1. Rule-based detection
    print("Computing NDVI and BSI...")
    rule_mask = detect_mining_rules(
        Path(args.image),
        ndvi_threshold=ndvi_threshold,
        bsi_threshold=bsi_threshold,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rule_mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Rule-Based Baseline Mask")
    ax.axis("off")
    fig.savefig(mask_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline mask to {mask_path}")

    # 2. Convert rule mask to polygons
    with rasterio.open(args.image) as src:
        transform = src.transform
        crs = src.crs

    gdf_rules = rules_to_polygons(rule_mask, transform, crs)
    gdf_rules.to_file(polygons_path, driver="GPKG")
    rule_count = len(gdf_rules)
    print(f"Rule-based detections: {rule_count} polygons")

    # 3. Ground truth
    gdf_labels = load_labels(args.labels, target_crs=str(crs))
    gt_mask = burn_mask(gdf_labels, args.image).astype(bool)
    gt_count = len(gdf_labels)

    # 4. AI inference
    print("Running AI model inference...")
    tiles = tile_image(args.image, tile_size=tile_size, overlap=overlap)
    if not tiles:
        raise ValueError(
            f"Image {args.image} is smaller than tile_size ({tile_size}) "
            "and cannot be tiled."
        )
    first_tile, _ = tiles[0]
    in_channels = first_tile.shape[0]

    predict_big_image(
        args.image,
        args.model,
        ai_pred_path,
        tile_size=tile_size,
        overlap=overlap,
        in_channels=in_channels,
        device=args.device,
    )
    print("AI blending complete.")

    # 5. AI polygons
    ai_gdf = postprocess(ai_pred_path, ai_polygons_path, threshold=threshold)
    ai_count = len(ai_gdf)
    print(f"AI model detections: {ai_count} polygons")

    # 6. Metrics
    rule_iou = compute_iou(
        rule_mask.astype(np.float32),
        gt_mask.astype(np.float32),
        threshold=0.5,
    )
    ai_metrics = evaluate_prediction(
        ai_pred_path,
        Path(args.labels),
        threshold=threshold,
    )
    ai_iou = ai_metrics["iou"]

    print(f"Ground truth labels: {gt_count} polygons")
    print(f"Rule IoU vs GT: {rule_iou:.2f}")
    print(f"AI IoU vs GT: {ai_iou:.2f}")

    if rule_iou > 0:
        improvement = ((ai_iou - rule_iou) / rule_iou) * 100
        if improvement > 0:
            print(f"✅ AI beats baseline by +{improvement:.0f}%")
        else:
            print(f"⚠️  AI underperforms baseline by {improvement:.0f}%")
    else:
        if ai_iou > 0:
            print("✅ AI beats baseline (baseline IoU is 0)")
        else:
            print("⚠️  Both AI and baseline have IoU of 0")

    # 7. Comparison figure
    create_comparison_figure(
        Path(args.image),
        ai_pred_path,
        rule_mask,
        Path(args.labels),
        comparison_path,
        threshold=threshold,
    )
    print(f"Saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
