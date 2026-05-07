"""Training script for semantic segmentation.

Supports two modes:
  1. Single synthetic image (Milestone 3):
       python -m goldmine_watch.training.train --fake <image.tif> <labels.gpkg>
  2. Patches on disk (Milestone 7):
       python -m goldmine_watch.training.train --patches <patches_dir>
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from goldmine_watch.data.dataset import PatchDataset
from goldmine_watch.models.unet import get_model


def train_patches(
    patches_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.001,
    device: str = "cpu",
) -> None:
    """Train the model on patches saved to disk.

    Args:
        patches_dir: Directory with image_*.npy and mask_*.npy files.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Device to train on.
    """
    device_obj = torch.device(device)
    dataset = PatchDataset(patches_dir, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Infer input channels from first sample
    sample_image, _ = dataset[0]
    in_channels = sample_image.shape[0]

    model = get_model(in_channels=in_channels).to(device_obj)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for images, masks in loader:
            images = images.to(device_obj)
            masks = masks.to(device_obj)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

        # Save checkpoint every epoch
        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)

    print(f"Training complete. Checkpoints saved to {checkpoint_dir}/")


def train_fake(
    image_path: str | Path,
    labels_path: str | Path,
    epochs: int = 5,
    lr: float = 0.001,
    device: str = "cpu",
    output_dir: str | Path = "models",
) -> None:
    """Train on a single synthetic image for Milestone 3.

    Args:
        image_path: Path to synthetic GeoTIFF.
        labels_path: Path to synthetic labels.
        epochs: Number of epochs.
        lr: Learning rate.
        device: Device to train on.
        output_dir: Directory to save the checkpoint.
    """
    import numpy as np
    import rasterio

    from goldmine_watch.data.ingest import burn_mask, load_labels

    device_obj = torch.device(device)
    gdf = load_labels(labels_path)

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)

    mask = burn_mask(gdf, image_path).astype(np.float32)

    image_t = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    in_channels = image.shape[0]
    model = get_model(in_channels=in_channels).to(device_obj)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        image_batch = image_t.to(device_obj)
        mask_batch = mask_t.to(device_obj)

        optimizer.zero_grad()
        outputs = model(image_batch)
        loss = criterion(outputs, mask_batch)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{epochs} — Loss: {loss.item():.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "milestone3_fake.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main() -> None:
    """Parse CLI arguments and dispatch training."""
    parser = argparse.ArgumentParser(description="Train GoldMine Watch segmentation model")
    parser.add_argument(
        "--fake", nargs=2, metavar=("IMAGE", "LABELS"), help="Train on a single synthetic image"
    )
    parser.add_argument("--patches", metavar="DIR", help="Train on patches directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for patch training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if args.fake:
        train_fake(args.fake[0], args.fake[1], epochs=args.epochs, lr=args.lr, device=args.device)
    elif args.patches:
        train_patches(
            args.patches,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
