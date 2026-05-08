#!/usr/bin/env python3
"""Demo script for Feature 4: train on patches with metrics and spatial validation.

Usage:
    python scripts/demo_feature4_train.py outputs/patches --epochs 30 --device cuda
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from goldmine_watch.training.train import train_patches


def plot_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    """Save loss and IoU curves as PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # IoU curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_iou"], label="Val IoU", color="green", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Validation IoU")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "iou_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse arguments and run training demo."""
    parser = argparse.ArgumentParser(description="Demo training with metrics")
    parser.add_argument("patches_dir", help="Directory with training patches")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument(
        "--output-dir",
        default="outputs/training",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="models",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()

    history = train_patches(
        patches_dir=args.patches_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.checkpoint_dir,
    )

    plot_curves(history, Path(args.output_dir))
    print(f"Plots saved to {args.output_dir}/")
    print("  - loss_curve.png")
    print("  - iou_curve.png")


if __name__ == "__main__":
    main()
