"""Standalone script to plot mining labels on a simple map.

Usage:
    python scripts/plot_labels.py <labels.gpkg> [output.png]
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


def plot_labels(labels_path: str | Path, output_path: str | Path | None = None) -> None:
    """Load labels and plot them on a map with basic statistics.

    Args:
        labels_path: Path to the vector label file.
        output_path: Optional path to save the plot PNG.
    """
    labels_path = Path(labels_path)
    gdf = gpd.read_file(labels_path)

    # Drop empty/invalid for display
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]

    print(f"Labels file: {labels_path}")
    print(f"  Polygons: {len(gdf)}")
    if gdf.crs is not None:
        print(f"  CRS: {gdf.crs}")
    if not gdf.empty:
        area_m2 = gdf.to_crs(gdf.estimate_utm_crs()).area.sum()
        print(f"  Total area: {area_m2:,.0f} m² ({area_m2 / 10_000:.2f} ha)")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color="red", edgecolor="black", alpha=0.5)
    ax.set_title("Mining Surface Labels")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_labels.py <labels.gpkg> [output.png]")
        sys.exit(1)

    labels = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    plot_labels(labels, out)
