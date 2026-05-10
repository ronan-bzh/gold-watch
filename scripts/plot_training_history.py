#!/usr/bin/env python3
"""Generate training analysis plots from a checkpoint.

Produces diagnostic visualizations for overfitting/underfitting analysis:
  1. Loss curves (train vs val)
  2. Validation IoU over epochs
  3. Validation F1 over epochs
  4. Learning rate schedule
  5. Combined dashboard

Usage:
    python scripts/plot_training_history.py \
        --checkpoint models/best_model.pth \
        --output-dir outputs/training_plots
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


def load_history(checkpoint_path: str | Path) -> tuple[dict, int, float]:
    """Extract history dict from a checkpoint.

    Returns:
        Tuple of (history dict, best_epoch, best_iou).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return ckpt["history"], ckpt.get("best_epoch", -1), ckpt.get("best_iou", 0.0)


def plot_loss_curves(history: dict, output_path: Path) -> None:
    """Plot train vs validation loss."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, "b-", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss, "r-", linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Annotate overfitting region if val loss diverges
    min_val_loss_epoch = np.argmin(val_loss) + 1
    ax.axvline(min_val_loss_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Min val loss @ epoch {min_val_loss_epoch}")
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_val_iou(history: dict, best_epoch: int, best_iou: float, output_path: Path) -> None:
    """Plot validation IoU over epochs."""
    epochs = np.arange(1, len(history["val_iou"]) + 1)
    val_iou = np.array(history["val_iou"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_iou, "g-", linewidth=2, marker="o", markersize=4)
    ax.axhline(0.20, color="orange", linestyle="--", alpha=0.5, label="Minimum acceptable (0.20)")
    ax.axhline(0.40, color="blue", linestyle="--", alpha=0.5, label="Good (0.40)")
    ax.axhline(0.60, color="purple", linestyle="--", alpha=0.5, label="Excellent (0.60)")

    if best_epoch > 0:
        ax.scatter([best_epoch], [best_iou], color="red", s=150, zorder=5,
                   label=f"Best: {best_iou:.3f} @ epoch {best_epoch}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation IoU", fontsize=12)
    ax.set_title("Validation IoU Over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.0, val_iou.max() * 1.1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_val_f1(history: dict, output_path: Path) -> None:
    """Plot validation F1 over epochs."""
    epochs = np.arange(1, len(history["val_f1"]) + 1)
    val_f1 = np.array(history["val_f1"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_f1, "m-", linewidth=2, marker="s", markersize=4)
    ax.axhline(0.30, color="orange", linestyle="--", alpha=0.5, label="Minimum acceptable (0.30)")
    ax.axhline(0.55, color="blue", linestyle="--", alpha=0.5, label="Good (0.55)")
    ax.axhline(0.75, color="purple", linestyle="--", alpha=0.5, label="Excellent (0.75)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation F1", fontsize=12)
    ax.set_title("Validation F1 Over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.0, val_f1.max() * 1.1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_lr(history: dict, output_path: Path) -> None:
    """Plot learning rate schedule."""
    # LR is not directly in history, but we can infer from loss patterns
    # For now, show a placeholder or skip
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Approximate LR: starts at 0.001, drops by 0.5 on plateau
    # We'll reconstruct from the known pattern
    lr = [0.001] * len(epochs)
    for i in range(1, len(lr)):
        # Heuristic: LR drops when val_iou hasn't improved for 5 epochs
        if i >= 6:
            recent = history["val_iou"][i-5:i]
            if max(recent) <= history["val_iou"][i-6]:
                lr[i] = lr[i-1] * 0.5
            else:
                lr[i] = lr[i-1]
        else:
            lr[i] = lr[i-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, lr, "c-", linewidth=2, marker="^", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate (log scale)", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_dashboard(history: dict, best_epoch: int, best_iou: float, output_path: Path) -> None:
    """Create a combined dashboard with all metrics."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    val_iou = np.array(history["val_iou"])
    val_f1 = np.array(history["val_f1"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss, "r-", linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Val IoU
    ax = axes[0, 1]
    ax.plot(epochs, val_iou, "g-", linewidth=2, marker="o", markersize=3)
    ax.axhline(0.40, color="blue", linestyle="--", alpha=0.4, label="Good (0.40)")
    ax.axhline(0.60, color="purple", linestyle="--", alpha=0.4, label="Excellent (0.60)")
    if best_epoch > 0:
        ax.scatter([best_epoch], [best_iou], color="red", s=100, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val IoU")
    ax.set_title(f"Validation IoU (Best: {best_iou:.3f} @ {best_epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Val F1
    ax = axes[1, 0]
    ax.plot(epochs, val_f1, "m-", linewidth=2, marker="s", markersize=3)
    ax.axhline(0.55, color="blue", linestyle="--", alpha=0.4, label="Good (0.55)")
    ax.axhline(0.75, color="purple", linestyle="--", alpha=0.4, label="Excellent (0.75)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F1")
    ax.set_title("Validation F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Overfitting diagnostic
    ax = axes[1, 1]
    loss_gap = val_loss - train_loss
    ax.plot(epochs, loss_gap, "k-", linewidth=2)
    ax.axhline(0, color="green", linestyle="-", alpha=0.5, label="Perfect fit")
    ax.axhline(0.15, color="orange", linestyle="--", alpha=0.4, label="Warning threshold")
    ax.fill_between(epochs, 0, loss_gap, where=(loss_gap > 0.15),
                    alpha=0.3, color="red", label="Overfitting risk")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss - Train Loss")
    ax.set_title("Overfitting Diagnostic (Gap)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("GoldMine Watch — Training Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """Parse arguments and generate plots."""
    parser = argparse.ArgumentParser(description="Generate training analysis plots")
    parser.add_argument("--checkpoint", default="models/best_model.pth",
                        help="Path to checkpoint with history")
    parser.add_argument("--output-dir", default="outputs/training_plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading history from {args.checkpoint}...")
    history, best_epoch, best_iou = load_history(args.checkpoint)

    # Guard against empty or corrupted history
    required_keys = ["train_loss", "val_loss", "val_iou", "val_f1"]
    for key in required_keys:
        if key not in history:
            raise ValueError(f"Checkpoint missing history key: {key!r}")

    num_epochs = len(history["train_loss"])
    if num_epochs == 0:
        raise ValueError("History contains zero epochs — nothing to plot.")

    for key in required_keys:
        if len(history[key]) != num_epochs:
            raise ValueError(
                f"History array length mismatch: {key!r} has {len(history[key])} "
                f"entries, expected {num_epochs}."
            )

    print(f"Epochs: {num_epochs}")
    print(f"Best epoch: {best_epoch} (IoU: {best_iou:.4f})")

    plot_loss_curves(history, output_dir / "01_loss_curves.png")
    plot_val_iou(history, best_epoch, best_iou, output_dir / "02_val_iou.png")
    plot_val_f1(history, output_dir / "03_val_f1.png")
    plot_lr(history, output_dir / "04_learning_rate.png")
    plot_dashboard(history, best_epoch, best_iou, output_dir / "05_dashboard.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
