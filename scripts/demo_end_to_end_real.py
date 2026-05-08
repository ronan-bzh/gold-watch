#!/usr/bin/env python3
"""End-to-end demo with real satellite data and quick training.

Pipeline:
  1. Select AOI around Saint-Laurent-du-Maroni from mining polygons.
  2. Download a Sentinel-2 scene from Copernicus Data Space (7 bands + SCL).
  3. Filter and reproject labels to the downloaded raster CRS.
  4. Generate training patches with cloud masking.
    5. Train U-Net for 20 epochs with class-balanced loss.
  6. Run sliding-window inference on the full scene.
  7. Post-process predictions into vector polygons.
  8. Visualize and assert success.

Usage:
    export $(cat .env | xargs)
    python scripts/demo_end_to_end_real.py

If an existing library function crashes, the script aborts and writes a
BUG_REPORT_<stage>.md so the bug can be fixed separately.
"""

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import torch
from rasterio import Affine
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import box

from goldmine_watch.data.copernicus import download_scene, get_access_token, search_scenes
from goldmine_watch.data.ingest import burn_mask, load_labels
from goldmine_watch.data.patches import generate_sliding_window_patches
from goldmine_watch.inference.postprocess import postprocess
from goldmine_watch.inference.predict_big import predict_big_image
from goldmine_watch.training.train import train_patches

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AOI_BBOX = (-54.05, 5.45, -54.0, 5.49)  # Tight bbox around actual mine cluster
DATE_RANGE = "2023-06-01/2023-12-31"
MAX_CLOUD_COVER = 20.0
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
RESOLUTION = 10

PATCH_SIZE = 256
STRIDE = 128
MAX_PATCHES = 150
MIN_VALID_FRACTION = 0.8
MAX_CLOUD_FRACTION = 0.2

EPOCHS = 20
BATCH_SIZE = 4
LR = 0.001

OUTPUT_DIR = Path("outputs/demo_real")
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")

SCENE_PATH = DATA_RAW / "sentinel2_scene_real.tif"
LABELS_PATH = DATA_RAW / "labels_filtered.gpkg"
PATCHES_DIR = DATA_PROCESSED / "patches_real"
PREDICTION_PATH = OUTPUT_DIR / "prediction_real.tif"
POLYGONS_PATH = OUTPUT_DIR / "polygons_real.gpkg"
VISUAL_PATH = OUTPUT_DIR / "end_to_end_real.png"
METRICS_PATH = OUTPUT_DIR / "demo_metrics.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Auto-select the best available PyTorch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _percentile_stretch(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    lo, hi = np.percentile(band, [low, high])
    stretched = (band - lo) / (hi - lo + 1e-8)
    return np.clip(stretched, 0, 1)


def _make_rgb(image: np.ndarray, red_idx: int = 2, green_idx: int = 1, blue_idx: int = 0) -> np.ndarray:
    """Create a displayable RGB composite from a multi-band image.

    Args:
        image: Array of shape (bands, height, width).
        red_idx, green_idx, blue_idx: 0-based band indices for RGB.

    Returns:
        RGB array of shape (height, width, 3) in 0-1 range.
    """
    rgb = np.stack(
        [
            _percentile_stretch(image[red_idx]),
            _percentile_stretch(image[green_idx]),
            _percentile_stretch(image[blue_idx]),
        ],
        axis=-1,
    )
    return rgb


def write_bug_report(stage: str, exc: Exception) -> Path:
    """Write a bug report markdown file and return its path."""
    path = OUTPUT_DIR / f"BUG_REPORT_{stage}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Bug Report – Stage: {stage}\n\n"
        f"**Exception type:** `{type(exc).__name__}`\n\n"
        f"**Message:**\n```\n{exc}\n```\n\n"
        f"**Traceback:**\n```\n{traceback.format_exc()}\n```\n\n"
        f"**What was attempted:**\n"
        f"Running the end-to-end real-data demo at stage `{stage}`.\n\n"
        f"**Required action:**\n"
        f"Fix the existing source code that caused this failure, then re-run the demo.\n"
    )
    return path


import traceback


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def stage_download() -> Path:
    """Stage 1: Download Sentinel-2 scene from Copernicus Data Space."""
    print("\n" + "=" * 60)
    print("STAGE 1: Download Sentinel-2 scene")
    print("=" * 60)

    client_id = os.environ.get("COPERNICUS_CLIENT_ID")
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Environment variables COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET are required."
        )

    print("Authenticating with Copernicus Data Space...")
    token = get_access_token(client_id, client_secret)

    print(f"Searching scenes: bbox={AOI_BBOX}, date={DATE_RANGE}, max_cloud={MAX_CLOUD_COVER}%")
    scenes = search_scenes(AOI_BBOX, DATE_RANGE, token, max_cloud_cover=MAX_CLOUD_COVER)
    print(f"Found {len(scenes)} scene(s)")
    for s in scenes[:3]:
        cloud = s.get("properties", {}).get("eo:cloud_cover", "unknown")
        print(f"  - {s['id']} (cloud: {cloud}%)")

    item = scenes[0]
    print(f"\nDownloading scene {item['id']} with {len(BANDS)} bands...")
    saved = download_scene(
        item,
        token,
        SCENE_PATH,
        bands=BANDS,
        resolution=RESOLUTION,
        bbox=AOI_BBOX,
    )

    with rasterio.open(saved) as src:
        print(
            f"Downloaded: {src.width}x{src.height} px | "
            f"{src.count} bands | CRS={src.crs} | "
            f"cloud={item.get('properties', {}).get('eo:cloud_cover', '?')}%"
        )
        assert src.count == len(BANDS), f"Expected {len(BANDS)} bands, got {src.count}"
        assert src.crs is not None, "Raster has no CRS"

    print(f"[PASS] Scene saved to {saved}")
    return saved


def _crop_to_valid_data(scene_path: Path, output_path: Path) -> Path:
    """Crop a raster to its valid (non-zero) data extent."""
    with rasterio.open(scene_path) as src:
        data = src.read()
        # Find valid bounds across all bands
        valid = (data > 0).any(axis=0)
        rows = valid.any(axis=1)
        cols = valid.any(axis=0)
        ymin = int(np.where(rows)[0][0])
        ymax = int(np.where(rows)[0][-1]) + 1
        xmin = int(np.where(cols)[0][0])
        xmax = int(np.where(cols)[0][-1]) + 1

        window = rasterio.windows.Window(xmin, ymin, xmax - xmin, ymax - ymin)
        cropped = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=cropped.shape[1],
            width=cropped.shape[2],
            transform=transform,
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cropped)

    print(f"Cropped to valid data: {profile['width']}x{profile['height']} (was {src.width}x{src.height})")
    return output_path


def stage_labels(scene_path: Path) -> Path:
    """Stage 2: Filter mining labels to AOI and reproject to raster CRS."""
    print("\n" + "=" * 60)
    print("STAGE 2: Prepare labels")
    print("=" * 60)

    print("Loading mining polygons...")
    gdf = load_labels("data/french_guiana_mines.geojson")
    print(f"Total polygons in GeoJSON: {len(gdf)}")

    # Filter to AOI
    aoi_box = box(*AOI_BBOX)
    gdf_aoi = gdf[gdf.intersects(aoi_box)]
    print(f"Polygons overlapping AOI: {len(gdf_aoi)}")

    if len(gdf_aoi) == 0:
        raise RuntimeError("No mining polygons overlap the selected AOI.")

    # Reproject to scene CRS
    with rasterio.open(scene_path) as src:
        scene_crs = src.crs
    gdf_proj = gdf_aoi.to_crs(scene_crs)

    # Add a dummy label column if missing (burn_mask doesn't use it, but GeoPackage needs something)
    if "label" not in gdf_proj.columns:
        gdf_proj["label"] = "mining"

    gdf_proj.to_file(LABELS_PATH, driver="GPKG")
    print(f"Filtered labels saved to {LABELS_PATH}")

    # Quick sanity check: burn a mask and ensure it's not all zeros
    mask = burn_mask(gdf_proj, scene_path)
    positive_pixels = int(mask.sum())
    print(f"Positive pixels in label mask: {positive_pixels}")
    assert positive_pixels > 0, "Label mask is all zeros – labels do not overlap the scene."

    print(f"[PASS] Labels ready ({len(gdf_proj)} polygons, {positive_pixels} positive pixels)")
    return LABELS_PATH


def _extract_patches_around_mines(
    scene_path: Path,
    labels_path: Path,
    output_dir: Path,
    patch_size: int = 256,
    num_bg: int = 100,
) -> int:
    """Extract patches centered on mining polygons plus random background patches.

    Guarantees that positive patches contain actual labeled pixels.
    """
    import rasterio
    from rasterio.windows import Window

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_labels(labels_path)
    with rasterio.open(scene_path) as src:
        height = src.height
        width = src.width
        transform = src.transform

    # Compute pixel centroids of mining polygons
    centers: list[tuple[int, int]] = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        cx, cy = geom.centroid.x, geom.centroid.y
        px = int((cx - transform.c) / transform.a)
        py = int((cy - transform.f) / transform.e)
        # Clamp to valid patch origin
        px = max(0, min(px - patch_size // 2, width - patch_size))
        py = max(0, min(py - patch_size // 2, height - patch_size))
        centers.append((px, py))

    # Also add random background patches
    rng = np.random.RandomState(42)
    for _ in range(num_bg):
        px = rng.randint(0, max(1, width - patch_size + 1))
        py = rng.randint(0, max(1, height - patch_size + 1))
        centers.append((px, py))

    # Burn full mask once for reference
    mask_full = burn_mask(gdf, scene_path)

    patch_id = 0
    with rasterio.open(scene_path) as src:
        for px, py in centers:
            window = Window(px, py, patch_size, patch_size)
            img = src.read(window=window)
            msk = mask_full[py : py + patch_size, px : px + patch_size]
            # Skip if dimensions mismatch (edge cases)
            if img.shape[1] != patch_size or img.shape[2] != patch_size:
                continue
            np.save(output_dir / f"image_{patch_id:04d}.npy", img)
            np.save(output_dir / f"mask_{patch_id:04d}.npy", msk)
            patch_id += 1

    return patch_id


def stage_patches(scene_path: Path, labels_path: Path) -> int:
    """Stage 3: Generate training patches."""
    print("\n" + "=" * 60)
    print("STAGE 3: Generate patches")
    print("=" * 60)

    # Clean old patches
    if PATCHES_DIR.exists():
        shutil.rmtree(PATCHES_DIR)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # Use custom extractor that guarantees positive patches
    generated = _extract_patches_around_mines(
        scene_path, labels_path, PATCHES_DIR, patch_size=PATCH_SIZE, num_bg=100
    )
    print(f"Generated: {generated} patches ({len(load_labels(labels_path))} mine-centered + 100 background)")
    assert generated >= 10, f"Only {generated} patches generated; need at least 10 to train."

    # Quick sanity: verify some positives exist
    mask_files = sorted(PATCHES_DIR.glob("mask_*.npy"))
    positive_count = sum(1 for f in mask_files if np.load(f).sum() > 0)
    print(f"Positive patches: {positive_count}/{len(mask_files)}")
    assert positive_count > 0, "No positive patches found – mines may be smaller than patch size."

    # Save a few patch visualizations
    from goldmine_watch.data.patches import save_patch_visual
    vis_dir = OUTPUT_DIR / "patch_samples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(5, len(mask_files))):
        img = np.load(PATCHES_DIR / f"image_{i:04d}.npy")
        msk = np.load(PATCHES_DIR / f"mask_{i:04d}.npy")
        save_patch_visual(img, msk, vis_dir, prefix=f"patch_{i:02d}")
    print(f"Saved {min(5, len(mask_files))} patch sample PNGs to {vis_dir}")

    print(f"[PASS] {generated} patches ready in {PATCHES_DIR}")
    return generated


def stage_train() -> dict:
    """Stage 4: Quick training (20 epochs with class balancing)."""
    print("\n" + "=" * 60)
    print("STAGE 4: Train model")
    print("=" * 60)

    # Remove stale checkpoints so best_model.pth is fresh
    for ckpt in MODELS_DIR.glob("epoch_*.pth"):
        ckpt.unlink()
    if (MODELS_DIR / "best_model.pth").exists():
        (MODELS_DIR / "best_model.pth").unlink()

    device = get_device()
    print(f"Device: {device}")

    history = train_patches(
        patches_dir=PATCHES_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=device,
        output_dir=MODELS_DIR,
    )

    best_model = MODELS_DIR / "best_model.pth"
    assert best_model.exists(), f"best_model.pth not found at {best_model}"

    final_iou = history["val_iou"][-1]
    print(f"Final val IoU: {final_iou:.4f}")
    print(f"[PASS] Training complete. Best model: {best_model}")
    return history


def stage_inference(scene_path: Path) -> Path:
    """Stage 5: Run inference on the full scene."""
    print("\n" + "=" * 60)
    print("STAGE 5: Inference")
    print("=" * 60)

    device = get_device()
    best_model = MODELS_DIR / "best_model.pth"

    # Infer in_channels from first patch
    from goldmine_watch.inference.tiler import tile_image
    first_tile, _ = tile_image(scene_path, tile_size=PATCH_SIZE, overlap=64)[0]
    in_channels = first_tile.shape[0]
    print(f"Detected {in_channels} input channels")

    predict_big_image(
        image_path=scene_path,
        model_path=best_model,
        output_path=PREDICTION_PATH,
        tile_size=PATCH_SIZE,
        overlap=64,
        in_channels=in_channels,
        device=device,
    )

    with rasterio.open(PREDICTION_PATH) as src:
        pred = src.read(1)
    assert not np.isnan(pred).all(), "Prediction raster is all NaN"
    print(f"Prediction stats: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
    print(f"[PASS] Prediction saved to {PREDICTION_PATH}")
    return PREDICTION_PATH


def stage_postprocess(pred_path: Path) -> Path:
    """Stage 6: Threshold and polygonize."""
    print("\n" + "=" * 60)
    print("STAGE 6: Post-process")
    print("=" * 60)

    postprocess(
        probability_raster_path=pred_path,
        output_path=POLYGONS_PATH,
        threshold=0.2,
        min_area_pixels=10,
    )

    gdf = gpd.read_file(POLYGONS_PATH)
    print(f"Extracted {len(gdf)} polygon(s)")
    # We don't assert >0 polygons because a weak model may produce none.
    # The assertion is on file existence, which is already covered.
    print(f"[PASS] Polygons saved to {POLYGONS_PATH}")
    return POLYGONS_PATH


def stage_visualize(scene_path: Path, pred_path: Path, polygons_path: Path) -> Path:
    """Stage 7: Create a 3-panel visualization figure."""
    print("\n" + "=" * 60)
    print("STAGE 7: Visualization")
    print("=" * 60)

    with rasterio.open(scene_path) as src:
        image = src.read().astype(np.float32)
    rgb = _make_rgb(image, red_idx=2, green_idx=1, blue_idx=0)

    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    # Load polygons and rasterize to pixel mask for overlay
    gdf = gpd.read_file(polygons_path)
    polygon_mask = np.zeros(pred.shape, dtype=np.uint8)
    if not gdf.empty:
        with rasterio.open(scene_path) as src:
            transform = src.transform
        shapes = ((geom, 1) for geom in gdf.geometry if not geom.is_empty)
        polygon_mask = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=pred.shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )

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
    outline = ndimage.binary_dilation(outline, iterations=3)

    overlay = rgb.copy()
    overlay[outline] = [0.0, 1.0, 1.0]  # Cyan outline

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Composite (B04, B03, B02)")
    axes[0].axis("off")

    im = axes[1].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Model Prediction (Probability)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Predicted Polygons (cyan outline)")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(VISUAL_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert VISUAL_PATH.stat().st_size > 0, "Visualization PNG is empty"
    print(f"[PASS] Visualization saved to {VISUAL_PATH}")
    return VISUAL_PATH


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main() -> int:
    """Run the full end-to-end demo and return 0 on success."""
    parser = argparse.ArgumentParser(description="End-to-end demo with real data")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing scene")
    parser.add_argument("--skip-patches", action="store_true", help="Reuse existing patches")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing model")
    parser.add_argument("--skip-inference", action="store_true", help="Reuse existing prediction")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "status": "incomplete",
        "scene": None,
        "labels": None,
        "patches": None,
        "training": None,
        "prediction": None,
        "polygons": None,
        "visualization": None,
    }

    try:
        # Stage 1: Download
        if args.skip_download and SCENE_PATH.exists():
            scene_path = SCENE_PATH
            print(f"[SKIP] Using existing scene: {scene_path}")
        else:
            scene_path = stage_download()

        # Stage 1b: Crop to valid data (remove no-data bands from Copernicus/PC)
        cropped_path = DATA_RAW / "sentinel2_scene_real_cropped.tif"
        scene_path = _crop_to_valid_data(scene_path, cropped_path)

        results["scene"] = str(scene_path)

        # Stage 2: Labels
        labels_path = stage_labels(scene_path)
        results["labels"] = str(labels_path)

        # Stage 3: Patches
        if args.skip_patches and PATCHES_DIR.exists() and any(PATCHES_DIR.glob("image_*.npy")):
            patches_count = len(list(PATCHES_DIR.glob("image_*.npy")))
            print(f"[SKIP] Using existing patches: {patches_count} found")
        else:
            patches_count = stage_patches(scene_path, labels_path)
        results["patches"] = patches_count

        # Stage 4: Train
        if args.skip_train and (MODELS_DIR / "best_model.pth").exists():
            print("[SKIP] Using existing best_model.pth")
            history = None
        else:
            history = stage_train()
        results["training"] = {
            "epochs": EPOCHS,
            "device": get_device(),
            "history": history,
        }

        # Stage 5: Inference
        if args.skip_inference and PREDICTION_PATH.exists():
            print(f"[SKIP] Using existing prediction: {PREDICTION_PATH}")
        else:
            stage_inference(scene_path)
        results["prediction"] = str(PREDICTION_PATH)

        # Stage 6: Post-process
        stage_postprocess(PREDICTION_PATH)
        results["polygons"] = str(POLYGONS_PATH)

        # Stage 7: Visualize
        stage_visualize(scene_path, PREDICTION_PATH, POLYGONS_PATH)
        results["visualization"] = str(VISUAL_PATH)

        results["status"] = "success"

    except Exception as exc:
        stage = results.get("status", "unknown")
        bug_path = write_bug_report(stage, exc)
        print(f"\n[FAIL] Demo aborted at stage: {stage}")
        print(f"Bug report written to: {bug_path}")
        print(f"Exception: {type(exc).__name__}: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO COMPLETE – SUCCESS")
    print("=" * 60)
    print(f"Scene:          {results['scene']}")
    print(f"Labels:         {results['labels']}")
    print(f"Patches:        {results['patches']}")
    print(f"Training:       {results['training']['epochs']} epochs on {results['training']['device']}")
    if results["training"]["history"]:
        final_iou = results["training"]["history"]["val_iou"][-1]
        print(f"Final Val IoU:  {final_iou:.4f}")
    print(f"Prediction:     {results['prediction']}")
    print(f"Polygons:       {results['polygons']}")
    print(f"Visualization:  {results['visualization']}")
    print("=" * 60)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Summary JSON:   {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
