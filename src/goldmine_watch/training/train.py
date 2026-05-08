"""Training script for semantic segmentation.

Supports two modes:
  1. Single synthetic image (Milestone 3):
       python -m goldmine_watch.training.train --fake <image.tif> <labels.gpkg>
  2. Patches on disk (Milestone 7):
       python -m goldmine_watch.training.train --patches <patches_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from goldmine_watch.data.dataset import PatchDataset
from goldmine_watch.data.splits import spatial_train_val_split
from goldmine_watch.models.unet import get_model


def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one validation epoch and return (loss, iou, f1).

    Metrics are computed globally across all validation batches
    for an unbiased estimate.
    """
    model.eval()
    total_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()

            pred_binary = (preds >= 0.5).astype(bool)
            target_binary = (targets >= 0.5).astype(bool)
            total_tp += int(np.logical_and(pred_binary, target_binary).sum())
            total_fp += int(np.logical_and(pred_binary, ~target_binary).sum())
            total_fn += int(np.logical_and(~pred_binary, target_binary).sum())

    avg_loss = total_loss / len(loader)
    denom_iou = total_tp + total_fp + total_fn
    iou = total_tp / denom_iou if denom_iou > 0 else 1.0
    denom_pr = total_tp + total_fp
    precision = total_tp / denom_pr if denom_pr > 0 else 0.0
    denom_re = total_tp + total_fn
    recall = total_tp / denom_re if denom_re > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return avg_loss, iou, f1


def train_patches(
    patches_dir: str | Path,
    val_patches_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.001,
    device: str = "cpu",
    output_dir: str | Path = "models",
) -> dict[str, list[float]]:
    """Train the model on patches saved to disk.

    Args:
        patches_dir: Directory with image_*.npy and mask_*.npy files.
        val_patches_dir: Optional separate validation patches directory.
            If None, a spatial train/val split is performed automatically.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Device to train on.
        output_dir: Directory to save checkpoints.

    Returns:
        Dictionary with keys ``train_loss``, ``val_loss``, ``val_iou``,
        ``val_f1`` mapping to lists of per-epoch values.
    """
    device_obj = torch.device(device)
    patches_dir = Path(patches_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build train/val file lists
    if val_patches_dir is not None:
        train_files = sorted(patches_dir.glob("image_*.npy"))
        val_files = sorted(Path(val_patches_dir).glob("image_*.npy"))
        if not train_files:
            raise ValueError(f"No image_*.npy files found in {patches_dir}")
        if not val_files:
            raise ValueError(f"No image_*.npy files found in {val_patches_dir}")
    else:
        train_files, val_files = spatial_train_val_split(patches_dir, val_ratio=0.2)

    train_dataset = PatchDataset(patches_dir, augment=True, image_files=train_files)
    val_dataset = PatchDataset(
        patches_dir if val_patches_dir is None else val_patches_dir,
        augment=False,
        image_files=val_files,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Infer input channels from first sample
    sample_image, _ = train_dataset[0]
    in_channels = sample_image.shape[0]

    model = get_model(in_channels=in_channels).to(device_obj)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1": [],
    }

    best_iou = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device_obj)
            masks = masks.to(device_obj)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        val_loss, val_iou, val_f1 = _validate(model, val_loader, criterion, device_obj)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{epochs} — "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val IoU: {val_iou:.2f} | "
            f"Val F1: {val_f1:.2f}"
        )

        # Save checkpoint every epoch
        ckpt_path = output_dir / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # Save best model based on val IoU
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            best_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)

    print(f"Training complete. Checkpoints saved to {output_dir}/")
    print(f"Best model saved at epoch {best_epoch} (Val IoU: {best_iou:.2f})")

    return history


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
    parser.add_argument("--val-patches", metavar="DIR", help="Validation patches directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for patch training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", default="models", help="Directory to save checkpoints")
    args = parser.parse_args()

    if args.fake:
        train_fake(args.fake[0], args.fake[1], epochs=args.epochs, lr=args.lr, device=args.device)
    elif args.patches:
        train_patches(
            args.patches,
            val_patches_dir=args.val_patches,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
