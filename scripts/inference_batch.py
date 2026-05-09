#!/usr/bin/env python3
"""CLI entry-point for batch inference (Feature 13).

Usage::

    python scripts/inference_batch.py \
      --model models/phase2_best.pth \
      --output-dir outputs/phase2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from goldmine_watch.inference.batch import inference_batch


def main() -> None:
    """Parse arguments and run batch inference."""
    parser = argparse.ArgumentParser(description="Batch Inference Engine (Feature 13)")
    parser.add_argument(
        "--model",
        default="models/phase2_best.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=None,
        help="List of tile IDs (default: all 5 French Guiana tiles)",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Tile cache directory",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Probability threshold for downstream use",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Sliding-window tile size in pixels (default: 256 or config)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap between adjacent tiles in pixels (default: 64 or config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='PyTorch device ("auto", "cuda", "mps", "cpu")',
    )
    parser.add_argument(
        "--config",
        default="configs/mvp.yaml",
        help="Path to pipeline configuration YAML",
    )
    args = parser.parse_args()

    # Optional config overrides
    config: dict = {}
    if Path(args.config).exists():
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    inference_cfg = config.get("inference", {})
    model_path = (
        args.model
        if args.model != "models/phase2_best.pth"
        else inference_cfg.get("model_path", "models/phase2_best.pth")
    )
    tile_size = (
        args.tile_size
        if args.tile_size is not None
        else inference_cfg.get("tile_size", 256)
    )
    overlap = (
        args.overlap
        if args.overlap is not None
        else inference_cfg.get("overlap", 64)
    )
    threshold = (
        args.threshold
        if args.threshold is not None
        else inference_cfg.get("threshold", 0.2)
    )
    output_dir = (
        args.output_dir
        if args.output_dir != "outputs/phase2"
        else inference_cfg.get("output_dir", "outputs/phase2")
    )
    device = args.device or inference_cfg.get("device", None)

    inference_batch(
        model_path=model_path,
        tile_list=args.tiles,
        cache_dir=args.cache_dir,
        output_dir=output_dir,
        threshold=threshold,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
    )


if __name__ == "__main__":
    main()
