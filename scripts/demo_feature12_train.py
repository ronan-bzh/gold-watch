#!/usr/bin/env python3
"""Demo script for Feature 12: Full Territory Training.

Usage:
    python scripts/demo_feature12_train.py \
      --train data/splits/train \
      --val data/splits/val \
      --epochs 50 \
      --batch-size 8
"""

import argparse

from goldmine_watch.training.train import train_phase2


def main() -> None:
    """Parse arguments and run the Feature 12 training demo."""
    parser = argparse.ArgumentParser(description="Demo: Full Territory Training")
    parser.add_argument(
        "--train",
        default="data/splits/train",
        help="Directory with training patches",
    )
    parser.add_argument(
        "--val",
        default="data/splits/val",
        help="Directory with validation patches",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="auto", help='Device ("auto", "cuda", "mps", "cpu")')
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()

    print("Full Territory Training")
    print("=======================")

    history = train_phase2(
        train_dir=args.train,
        val_dir=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )

    print(f"\nTraining complete. Checkpoints saved to {args.output_dir}/")
    print("Best model saved based on validation IoU")
    print(f"Final Val IoU: {history['val_iou'][-1]:.2f}")


if __name__ == "__main__":
    main()
