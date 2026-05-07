"""Run model inference on a single image patch and save visualization.

Milestone 8: Can I predict on one patch?
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from goldmine_watch.models.unet import get_model


def predict_patch(
    image_patch: np.ndarray,
    model_path: str | Path,
    in_channels: int = 9,
    device: str = "cpu",
) -> np.ndarray:
    """Run model forward pass on a single image patch.

    Args:
        image_patch: Array of shape (bands, height, width).
        model_path: Path to the model checkpoint.
        in_channels: Number of input channels.
        device: Device to run on.

    Returns:
        Prediction array of shape (height, width) with probabilities.
    """
    device_obj = torch.device(device)
    model = get_model(in_channels=in_channels).to(device_obj)
    model.load_state_dict(torch.load(model_path, map_location=device_obj, weights_only=True))
    model.eval()

    image_t = torch.from_numpy(image_patch.astype(np.float32)).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        logits = model(image_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    return probs


def save_prediction_visual(
    image_patch: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a side-by-side visualization of the original patch and prediction.

    Args:
        image_patch: Array of shape (bands, height, width).
        prediction: Array of shape (height, width) with probabilities.
        output_path: Where to save the PNG.

    Returns:
        Path to the saved PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # RGB composite
    rgb = image_patch[:3].transpose(1, 2, 0).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)

    # Prediction heatmap
    pred_uint8 = (prediction * 255).astype(np.uint8)
    pred_rgb = np.stack([pred_uint8] * 3, axis=-1)

    side_by_side = np.concatenate([rgb, pred_rgb], axis=1)
    img = Image.fromarray(side_by_side)
    img.save(output_path)
    return output_path


def main() -> None:
    """CLI entrypoint for predict."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict on one patch")
    parser.add_argument("image", help="Path to image .npy patch")
    parser.add_argument("model", help="Path to model checkpoint")
    parser.add_argument("--out", default="outputs/prediction.png", help="Output PNG path")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    image_patch = np.load(args.image)
    in_channels = image_patch.shape[0]
    pred = predict_patch(image_patch, args.model, in_channels=in_channels, device=args.device)
    save_prediction_visual(image_patch, pred, args.out)
    print(f"Prediction saved to {args.out}")


if __name__ == "__main__":
    main()
