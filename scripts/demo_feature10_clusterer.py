#!/usr/bin/env python3
r"""Demo script for Feature 10: Mine Clusterer.

Usage::

    python scripts/demo_feature10_clusterer.py

Output is printed to stdout and shows the number of mines per Sentinel-2 tile.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from goldmine_watch.data.mine_clusterer import (
    cluster_mines_by_tile,
    compute_coverage_km,
    load_mines,
)


def main() -> int:
    """Run the mine clusterer demo."""
    print("Mine Clusterer Demo")
    print("===================")

    mines_gdf = load_mines("data/french_guiana_mines.geojson")
    print(f"Loading {len(mines_gdf)} mining polygons...")
    print("Clustering by Sentinel-2 tile...")
    print()

    clusters = cluster_mines_by_tile(mines_gdf)
    print(f"Required tiles: {len(clusters)}")
    for tile_id, group in clusters.items():
        print(f"  {tile_id}: {len(group)} mines")

    total = sum(len(g) for g in clusters.values())
    tile_ids = list(clusters.keys())
    width_km, height_km = compute_coverage_km(tile_ids)

    print()
    print(f"Total: {total} mines across {len(clusters)} tiles")
    print(f"Coverage: ~{width_km:.0f} km x {height_km:.0f} km")

    return 0


if __name__ == "__main__":
    sys.exit(main())
