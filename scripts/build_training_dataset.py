"""CLI script to build the full multi-scene training dataset.

Defaults are loaded from ``configs/mvp.yaml`` so that the CLI stays in
sync with the centralised configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from goldmine_watch.data.build_training_dataset import build_training_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("configs/mvp.yaml")


def _load_cfg_defaults() -> dict[str, object]:
    """Return dataset defaults from the centralised config file."""
    if not _CONFIG_PATH.exists():
        logger.warning("Config file %s not found; using hard-coded defaults.", _CONFIG_PATH)
        return {}
    cfg = OmegaConf.load(_CONFIG_PATH)
    dataset = cfg.get("dataset", {})
    geospatial = cfg.get("geospatial", {})
    paths = cfg.get("paths", {})
    cache = cfg.get("cache", {})
    return {
        "mines": dataset.get("source_mines", "data/french_guiana_mines.geojson"),
        "output": paths.get("data_splits", "data/splits"),
        "background": dataset.get("background_per_tile", 100),
        "patch_size": geospatial.get("patch_size", 256),
        "train_ratio": dataset.get("train_val_ratio", [0.8, 0.2])[0],
        "random_seed": dataset.get("random_seed", 42),
        "cache_dir": cache.get("tiles_dir", "data/cache/tiles"),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and build the training dataset."""
    defaults = _load_cfg_defaults()

    parser = argparse.ArgumentParser(
        description="Build multi-scene training dataset from all mines."
    )
    parser.add_argument(
        "--mines",
        default=defaults.get("mines", "data/french_guiana_mines.geojson"),
        help="Path to mining polygons GeoJSON.",
    )
    parser.add_argument(
        "--output",
        default=defaults.get("output", "data/splits"),
        help="Output directory for train/ and val/ splits.",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=defaults.get("background", 100),
        help="Number of background patches per tile.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=defaults.get("patch_size", 256),
        help="Patch size in pixels.",
    )
    parser.add_argument(
        "--date-range",
        default="2023-06-01/2023-12-31",
        help="Date range for tile downloads (ISO 8601 interval).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=defaults.get("train_ratio", 0.8),
        help="Fraction of tiles for training.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=defaults.get("random_seed", 42),
        help="Random seed for splitting and background sampling.",
    )
    parser.add_argument(
        "--cache-dir",
        default=defaults.get("cache_dir", "data/cache/tiles"),
        help="Directory for tile cache.",
    )

    args = parser.parse_args(argv)

    stats = build_training_dataset(
        mines_geojson=args.mines,
        output_dir=args.output,
        num_background_per_tile=args.background,
        patch_size=args.patch_size,
        date_range=args.date_range,
        train_val_ratio=(args.train_ratio, 1.0 - args.train_ratio),
        random_seed=args.random_seed,
        cache_dir=args.cache_dir,
    )

    print("\nMulti-Scene Training Dataset")
    print("=" * 40)
    print(f"Tiles processed: {stats['num_tiles']}")
    print(f"  Train tiles: {', '.join(stats['train_tiles'])}")
    print(f"  Val tiles:   {', '.join(stats['val_tiles'])}")
    print()
    print(f"Total patches:   {stats['total_patches']}")
    print(f"  Train patches: {stats['train_patches']}")
    print(f"  Val patches:   {stats['val_patches']}")
    print()
    print(f"Positive patches: {stats['positive_patches']}")
    print(f"Negative patches: {stats['negative_patches']}")
    print()
    print(f"Saved to {stats['output_dir']}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
