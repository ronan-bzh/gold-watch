"""Unit tests for QGIS export modules."""

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.export.qgis import create_qgis_project_full

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


def _make_geojson(path: Path) -> Path:
    """Create a GeoJSON with one polygon."""
    polygon = Polygon(
        [
            (200_050.0, 500_050.0),
            (200_150.0, 500_050.0),
            (200_150.0, 500_150.0),
            (200_050.0, 500_150.0),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[polygon],
        crs=TARGET_CRS,
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def _make_geotiff(path: Path, bands: int = 1) -> Path:
    """Create a minimal GeoTIFF."""
    data = np.zeros((bands, 128, 128), dtype=np.float32)
    if bands == 1:
        data[0, 40:90, 40:90] = 0.8
    transform = from_origin(200_000.0, 500_000.0 + 128 * 10.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=bands,
        dtype=data.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(data[i], i + 1)
    return path


class TestQGISExportFull:
    """Tests for full-territory QGIS project export."""

    def test_project_file_created(self, tmp_path: Path) -> None:
        """Should create a .qgz file."""
        mosaic_path = tmp_path / "mosaic.tif"
        detections_path = tmp_path / "detections.geojson"
        labels_path = tmp_path / "labels.geojson"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(mosaic_path)
        _make_geojson(detections_path)
        _make_geojson(labels_path)

        result = create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        assert result.exists()
        assert result.suffix == ".qgz"
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert any(name.endswith(".qgs") for name in names)

    def test_project_contains_all_layers(self, tmp_path: Path) -> None:
        """Should have at least 3 layers: mosaic, detections, labels."""
        mosaic_path = tmp_path / "mosaic.tif"
        detections_path = tmp_path / "detections.geojson"
        labels_path = tmp_path / "labels.geojson"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(mosaic_path)
        _make_geojson(detections_path)
        _make_geojson(labels_path)

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_name = [n for n in zf.namelist() if n.endswith(".qgs")][0]
            xml_content = zf.read(qgs_name).decode("utf-8")

        assert "probability_heatmap" in xml_content
        assert "detections" in xml_content
        assert "labels" in xml_content
        # Count maplayer elements
        layer_count = xml_content.count("<maplayer")
        assert layer_count >= 3

    def test_project_contains_four_layers_with_image(self, tmp_path: Path) -> None:
        """Should have 4 layers when image_path is provided."""
        mosaic_path = tmp_path / "mosaic.tif"
        detections_path = tmp_path / "detections.geojson"
        labels_path = tmp_path / "labels.geojson"
        image_path = tmp_path / "image.tif"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(mosaic_path)
        _make_geotiff(image_path, bands=3)
        _make_geojson(detections_path)
        _make_geojson(labels_path)

        create_qgis_project_full(
            mosaic_path,
            detections_path,
            labels_path,
            project_path,
            image_path=image_path,
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_name = [n for n in zf.namelist() if n.endswith(".qgs")][0]
            xml_content = zf.read(qgs_name).decode("utf-8")

        assert "satellite_composite" in xml_content
        assert "probability_heatmap" in xml_content
        assert "detections" in xml_content
        assert "labels" in xml_content
        layer_count = xml_content.count("<maplayer")
        assert layer_count == 4

    def test_detections_layer_red_outline(self, tmp_path: Path) -> None:
        """Detections should render with red outline."""
        mosaic_path = tmp_path / "mosaic.tif"
        detections_path = tmp_path / "detections.geojson"
        labels_path = tmp_path / "labels.geojson"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(mosaic_path)
        _make_geojson(detections_path)
        _make_geojson(labels_path)

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_name = [n for n in zf.namelist() if n.endswith(".qgs")][0]
            xml_content = zf.read(qgs_name).decode("utf-8")

        # Find the detections layer and check its outline color
        det_start = xml_content.find("<layername>Detections</layername>")
        assert det_start != -1
        det_end = xml_content.find("</maplayer>", det_start)
        det_section = xml_content[det_start:det_end]
        assert "255,0,0,255" in det_section

    def test_labels_layer_green_outline(self, tmp_path: Path) -> None:
        """Labels should render with green outline."""
        mosaic_path = tmp_path / "mosaic.tif"
        detections_path = tmp_path / "detections.geojson"
        labels_path = tmp_path / "labels.geojson"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(mosaic_path)
        _make_geojson(detections_path)
        _make_geojson(labels_path)

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_name = [n for n in zf.namelist() if n.endswith(".qgs")][0]
            xml_content = zf.read(qgs_name).decode("utf-8")

        # Find the labels layer and check its outline color
        lbl_start = xml_content.find("<layername>Labels</layername>")
        assert lbl_start != -1
        lbl_end = xml_content.find("</maplayer>", lbl_start)
        lbl_section = xml_content[lbl_start:lbl_end]
        assert "0,255,0,255" in lbl_section

    def test_probability_layer_has_pseudocolor_renderer(self, tmp_path: Path) -> None:
        """Probability heatmap should use singlebandpseudocolor renderer."""
        mosaic_path = tmp_path / "mosaic.tif"
        detections_path = tmp_path / "detections.geojson"
        labels_path = tmp_path / "labels.geojson"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(mosaic_path)
        _make_geojson(detections_path)
        _make_geojson(labels_path)

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_name = [n for n in zf.namelist() if n.endswith(".qgs")][0]
            xml_content = zf.read(qgs_name).decode("utf-8")

        assert "singlebandpseudocolor" in xml_content
        assert "colorrampshader" in xml_content

    def test_relative_paths_for_portability(self, tmp_path: Path) -> None:
        """Paths inside the project should be relative to the output directory."""
        mosaic_path = tmp_path / "mosaic.tif"
        detections_path = tmp_path / "detections.geojson"
        labels_path = tmp_path / "labels.geojson"
        project_path = tmp_path / "project.qgz"

        _make_geotiff(mosaic_path)
        _make_geojson(detections_path)
        _make_geojson(labels_path)

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_name = [n for n in zf.namelist() if n.endswith(".qgs")][0]
            xml_content = zf.read(qgs_name).decode("utf-8")

        assert "mosaic.tif" in xml_content
        assert "detections.geojson" in xml_content
        assert "labels.geojson" in xml_content
