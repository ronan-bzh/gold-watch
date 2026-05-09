#!/usr/bin/env python3
"""Training script for Feature 12: Full Territory Training.

Trains a U-Net with ResNet-34 encoder on the complete patch dataset,
using class-balanced BCE loss and ReduceLROnPlateau scheduling.

Usage:
    python scripts/train_phase2.py \
        --train data/splits/train \
        --val data/splits/val \
        --epochs 50 \
        --batch-size 8 \
        --device auto
"""

import argparse

from goldmine_watch.training.train import train_phase2


def main() -> None:
    """Parse CLI arguments and run phase-2 training."""
    parser = argparse.ArgumentParser(
        description="Full Territory Training (Feature 12)"
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Directory with training patches",
    )
    parser.add_argument(
        "--val",
        help="Directory with validation patches (optional)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device ("auto", "cuda", "mps", or "cpu")',
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=100.0,
        help="Maximum positive class weight",
    )
    args = parser.parse_args()

    print("Full Territory Training")
    print("=======================")

    history = train_phase2(
        train_dir=args.train,
        val_dir=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        save_every=args.save_every,
        max_pos_weight=args.max_pos_weight,
    )

    print(f"\nTraining complete. Checkpoints saved to {args.output_dir}/")
    print(f"Final Val IoU: {history['val_iou'][-1]:.2f}")


if __name__ == "__main__":
    main()
