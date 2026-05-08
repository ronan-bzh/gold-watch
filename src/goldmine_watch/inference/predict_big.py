"""Run inference on a full large image using sliding window + blending.

Milestone 9: Can I predict on a big image?
"""

from pathlib import Path

import numpy as np
import rasterio
import torch

from goldmine_watch.inference.blender import blend_predictions, normalize_canvas
from goldmine_watch.inference.evaluate import evaluate_prediction
from goldmine_watch.inference.tiler import tile_image
from goldmine_watch.models.unet import get_model


def predict_big_image(
    image_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    tile_size: int = 256,
    overlap: int = 64,
    in_channels: int = 9,
    device: str = "cpu",
) -> Path:
    """Run sliding-window inference on a large image and save as GeoTIFF.

    Args:
        image_path: Path to the input GeoTIFF.
        model_path: Path to the model checkpoint.
        output_path: Where to save the probability GeoTIFF.
        tile_size: Tile size in pixels.
        overlap: Overlap between tiles.
        in_channels: Number of input channels.
        device: Device to run on.

    Returns:
        Path to the saved GeoTIFF.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)
    model = get_model(in_channels=in_channels).to(device_obj)
    model.load_state_dict(torch.load(model_path, map_location=device_obj, weights_only=True))
    model.eval()

    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()

    tiles = tile_image(image_path, tile_size=tile_size, overlap=overlap)
    if not tiles:
        raise ValueError(
            f"Image {image_path} is smaller than tile_size ({tile_size}) and cannot be tiled."
        )

    canvas = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for tile_array, window in tiles:
            x, y = window.col_off, window.row_off
            tile_t = torch.from_numpy(tile_array.astype(np.float32)).unsqueeze(0).to(device_obj)
            logits = model(tile_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            blend_predictions(canvas, counts, probs, x, y, tile_size)

    final = normalize_canvas(canvas, counts)

    # Update profile for single-band float output
    profile.update(
        count=1,
        dtype=final.dtype,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final, 1)

    print(f"Prediction saved to {output_path}")
    return output_path


def main() -> None:
    """CLI entrypoint for predict_big."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict on a big image")
    parser.add_argument("image", help="Input GeoTIFF")
    parser.add_argument("model", help="Model checkpoint")
    parser.add_argument("--out", default="outputs/prediction_big.tif", help="Output GeoTIFF")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against ground truth")
    parser.add_argument("--labels", help="Path to vector labels (required with --evaluate)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Evaluation threshold")
    args = parser.parse_args()

    if args.evaluate and not args.labels:
        parser.error("--labels is required when --evaluate is set")

    # Infer channels from first tile
    first_tile, _ = tile_image(args.image, tile_size=args.tile_size, overlap=args.overlap)[0]
    in_channels = first_tile.shape[0]

    predict_big_image(
        args.image,
        args.model,
        args.out,
        tile_size=args.tile_size,
        overlap=args.overlap,
        in_channels=in_channels,
        device=args.device,
    )

    if args.evaluate:
        metrics = evaluate_prediction(
            Path(args.out),
            Path(args.labels),
            threshold=args.threshold,
        )
        print(
            f"IoU: {metrics['iou']:.2f} | F1: {metrics['f1']:.2f} | "
            f"Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f}"
        )


if __name__ == "__main__":
    main()
