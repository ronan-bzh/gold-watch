"""Functional tests for Feature 18: QGIS Export — Full Territory.

These tests exercise the full QGIS project creation workflow for the full
territory mosaic, detections, and labels.
"""

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.export.qgis import create_qgis_project_full

TARGET_CRS = "EPSG:2972"

pytestmark = pytest.mark.integration


def _make_probability_raster(tmp_path: Path, size: int = 256) -> Path:
    """Create a synthetic probability raster with a bright square."""
    raster_path = tmp_path / "mosaic.tif"
    probs = np.zeros((size, size), dtype=np.float32)
    probs[50:150, 50:150] = 0.8
    transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype=probs.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        dst.write(probs, 1)

    return raster_path


def _make_detections_geojson(tmp_path: Path) -> Path:
    """Create a synthetic detections GeoJSON."""
    path = tmp_path / "detections.geojson"
    polygon = Polygon(
        [
            (200_000.0, 500_000.0),
            (200_100.0, 500_000.0),
            (200_100.0, 500_100.0),
            (200_000.0, 500_100.0),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {"confidence": [0.85], "area_m2": [10_000.0]},
        geometry=[polygon],
        crs=TARGET_CRS,
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def _make_labels_geojson(tmp_path: Path) -> Path:
    """Create a synthetic labels GeoJSON."""
    path = tmp_path / "labels.geojson"
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


class TestFeature18QGISFlow:
    """End-to-end QGIS export workflow tests."""

    def test_project_opens_and_has_layers(self, tmp_path: Path) -> None:
        """.qgz file should contain a valid .qgs with all layers."""
        mosaic_path = _make_probability_raster(tmp_path)
        detections_path = _make_detections_geojson(tmp_path)
        labels_path = _make_labels_geojson(tmp_path)
        project_path = tmp_path / "goldmine_watch_full.qgz"

        result = create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        assert result.exists()
        assert result.suffix == ".qgz"

        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            qgs_files = [n for n in names if n.endswith(".qgs")]
            assert len(qgs_files) == 1
            xml_content = zf.read(qgs_files[0]).decode("utf-8")

        assert "Probability Heatmap" in xml_content
        assert "Detections" in xml_content
        assert "Labels" in xml_content
        assert "singlebandpseudocolor" in xml_content

    def test_all_layers_visible(self, tmp_path: Path) -> None:
        """All layers should be defined in the project XML."""
        mosaic_path = _make_probability_raster(tmp_path)
        detections_path = _make_detections_geojson(tmp_path)
        labels_path = _make_labels_geojson(tmp_path)
        project_path = tmp_path / "project.qgz"

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_files = [n for n in zf.namelist() if n.endswith(".qgs")]
            xml_content = zf.read(qgs_files[0]).decode("utf-8")

        layer_count = xml_content.count("<maplayer")
        assert layer_count == 3

    def test_detections_and_labels_distinguishable(self, tmp_path: Path) -> None:
        """Detections should be red and labels should be green."""
        mosaic_path = _make_probability_raster(tmp_path)
        detections_path = _make_detections_geojson(tmp_path)
        labels_path = _make_labels_geojson(tmp_path)
        project_path = tmp_path / "project.qgz"

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_files = [n for n in zf.namelist() if n.endswith(".qgs")]
            xml_content = zf.read(qgs_files[0]).decode("utf-8")

        assert "255,0,0,255" in xml_content  # red
        assert "0,255,0,255" in xml_content  # green

    def test_project_with_optional_rgb_image(self, tmp_path: Path) -> None:
        """When image_path is provided, project should have 4 layers."""
        mosaic_path = _make_probability_raster(tmp_path)
        detections_path = _make_detections_geojson(tmp_path)
        labels_path = _make_labels_geojson(tmp_path)
        image_path = tmp_path / "composite.tif"
        project_path = tmp_path / "project.qgz"

        # Create a 3-band RGB image
        data = np.zeros((3, 256, 256), dtype=np.float32)
        transform = from_origin(200_000.0, 500_000.0 + 256 * 10.0, 10.0, 10.0)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=3,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            for i in range(3):
                dst.write(data[i], i + 1)

        create_qgis_project_full(
            mosaic_path,
            detections_path,
            labels_path,
            project_path,
            image_path=image_path,
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_files = [n for n in zf.namelist() if n.endswith(".qgs")]
            xml_content = zf.read(qgs_files[0]).decode("utf-8")

        assert "Satellite Composite" in xml_content
        layer_count = xml_content.count("<maplayer")
        assert layer_count == 4

    def test_zoom_to_extent_metadata(self, tmp_path: Path) -> None:
        """Project should contain layer definitions that QGIS can zoom to."""
        mosaic_path = _make_probability_raster(tmp_path)
        detections_path = _make_detections_geojson(tmp_path)
        labels_path = _make_labels_geojson(tmp_path)
        project_path = tmp_path / "project.qgz"

        create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        with zipfile.ZipFile(project_path, "r") as zf:
            qgs_files = [n for n in zf.namelist() if n.endswith(".qgs")]
            xml_content = zf.read(qgs_files[0]).decode("utf-8")

        # Each layer should have a datasource and srs element
        assert xml_content.count("<datasource>") >= 3
        assert xml_content.count("<srs") >= 3

    def test_empty_geojson_handled_gracefully(self, tmp_path: Path) -> None:
        """Empty GeoJSON files should still produce a valid project."""
        mosaic_path = _make_probability_raster(tmp_path)
        detections_path = tmp_path / "empty_detections.geojson"
        labels_path = tmp_path / "empty_labels.geojson"
        project_path = tmp_path / "project.qgz"

        # Create empty GeoJSON feature collections
        for path in (detections_path, labels_path):
            gdf = gpd.GeoDataFrame(
                {},
                geometry=[],
                crs=TARGET_CRS,
            )
            gdf.to_file(path, driver="GeoJSON")

        result = create_qgis_project_full(
            mosaic_path, detections_path, labels_path, project_path
        )

        assert result.exists()
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert any(n.endswith(".qgs") for n in names)
