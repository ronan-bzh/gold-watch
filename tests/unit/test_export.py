"""Unit tests for export modules."""

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.export.csv import export_polygon_metrics
from goldmine_watch.export.qgis import create_qgis_project

TARGET_CRS = "EPSG:2972"


def _make_polygons_gpkg(path: Path) -> Path:
    """Create a GeoPackage with one polygon and required attributes."""
    polygon = Polygon(
        [
            (200_000.0, 500_000.0),
            (200_100.0, 500_000.0),
            (200_100.0, 500_100.0),
            (200_000.0, 500_100.0),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {
            "detection_id": [1],
            "confidence": [0.85],
            "area_m2": [10_000.0],
            "area_ha": [1.0],
        },
        geometry=[polygon],
        crs=TARGET_CRS,
    )
    gdf.to_file(path, driver="GPKG")
    return path


def _make_geotiff(path: Path) -> Path:
    """Create a minimal GeoTIFF."""
    data = np.zeros((128, 128), dtype=np.float32)
    transform = from_origin(200_000.0, 500_000.0 + 128 * 10.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=1,
        dtype=data.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


class TestExport:
    """Tests for CSV and QGIS export."""

    def test_csv_has_all_columns(self, tmp_path: Path) -> None:
        """CSV should have detection_id, area_m2, area_ha, confidence."""
        gpkg_path = tmp_path / "polygons.gpkg"
        _make_polygons_gpkg(gpkg_path)
        csv_path = tmp_path / "polygons.csv"

        result = export_polygon_metrics(gpkg_path, csv_path)

        assert result.exists()
        df = pd.read_csv(result)
        assert list(df.columns) == ["detection_id", "area_m2", "area_ha", "confidence"]
        assert len(df) == 1
        assert df.iloc[0]["detection_id"] == 1
        assert df.iloc[0]["area_m2"] == 10_000.0
        assert df.iloc[0]["area_ha"] == 1.0
        assert df.iloc[0]["confidence"] == 0.85

    def test_qgis_project_file_created(self, tmp_path: Path) -> None:
        """.qgz file should exist after export and contain a .qgs file."""
        image_path = tmp_path / "image.tif"
        prediction_path = tmp_path / "prediction.tif"
        polygons_path = tmp_path / "polygons.gpkg"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(image_path)
        _make_geotiff(prediction_path)
        _make_polygons_gpkg(polygons_path)

        result = create_qgis_project(image_path, prediction_path, polygons_path, project_path)

        assert result.exists()
        assert result.suffix == ".qgz"
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert any(name.endswith(".qgs") for name in names)
