"""Export polygon metrics to CSV."""

from pathlib import Path

import geopandas as gpd


def export_polygon_metrics(polygons_path: Path, output_csv: Path) -> Path:
    """Export polygon attributes to a CSV file.

    Args:
        polygons_path: Path to the vector file (GeoPackage, GeoJSON, etc.).
        output_csv: Path to save the CSV.

    Returns:
        Path to the saved CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(polygons_path)

    # Ensure expected columns exist; fill with defaults if missing
    expected = ["detection_id", "area_m2", "area_ha", "confidence"]
    for col in expected:
        if col not in gdf.columns:
            gdf[col] = None

    df = gdf[["detection_id", "area_m2", "area_ha", "confidence"]].copy()
    df.to_csv(output_csv, index=False)
    return output_csv
