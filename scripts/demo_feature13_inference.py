#!/usr/bin/env python3
"""Demo script for Feature 13: Batch Inference Engine.

Usage:
    python scripts/demo_feature13_inference.py \
      --model models/phase2_best.pth \
      --output outputs/phase2
"""

import argparse

from goldmine_watch.inference.batch import inference_batch


def main() -> None:
    """Parse arguments and run the Feature 13 batch inference demo."""
    parser = argparse.ArgumentParser(description="Demo: Batch Inference Engine")
    parser.add_argument(
        "--model",
        default="models/phase2_best.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        default="outputs/phase2",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Tile cache directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device ("auto", "cuda", "mps", "cpu")',
    )
    args = parser.parse_args()

    inference_batch(
        model_path=args.model,
        tile_list=None,  # default to all 5 French Guiana tiles
        cache_dir=args.cache_dir,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
