"""Pytest fixtures for synthetic geospatial test data."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

# French Guiana RGFG95 / UTM zone 22N
TARGET_CRS = "EPSG:2972"

# Synthetic image parameters
IMAGE_SIZE = 512
NUM_BANDS = 7
RESOLUTION = 10.0
ORIGIN_X = 200_000.0
ORIGIN_Y = 500_000.0


def _image_bounds() -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) for the synthetic image."""
    max_x = ORIGIN_X + IMAGE_SIZE * RESOLUTION
    max_y = ORIGIN_Y + IMAGE_SIZE * RESOLUTION
    return (ORIGIN_X, ORIGIN_Y, max_x, max_y)


@pytest.fixture
def synthetic_geotiff(tmp_path: Path) -> Path:
    """Create a 512x512 synthetic 7-band GeoTIFF with valid CRS and transform.

    Bands 1-6 are spectral bands; band 7 is tagged as SCL for cloud masking.
    """
    geotiff_path = tmp_path / "synthetic_image.tif"

    transform = from_origin(ORIGIN_X, ORIGIN_Y + IMAGE_SIZE * RESOLUTION, RESOLUTION, RESOLUTION)
    data = np.random.randint(0, 10_000, size=(NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)

    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        count=NUM_BANDS,
        dtype=data.dtype,
        crs=TARGET_CRS,
        transform=transform,
    ) as dst:
        dst.write(data)

    # Tag the last band as SCL so load_scl_band can find it
    with rasterio.open(geotiff_path, "r+") as dst:
        dst.set_band_description(NUM_BANDS, "SCL")

    return geotiff_path


@pytest.fixture
def synthetic_labels(tmp_path: Path) -> Path:
    """Create a GeoPackage with 2-3 simple rectangles as mining areas."""
    labels_path = tmp_path / "synthetic_labels.gpkg"

    min_x, min_y, max_x, max_y = _image_bounds()
    # Create 2-3 rectangles inside the image bounds
    polygons = [
        Polygon(
            [
                (min_x + 50, min_y + 50),
                (min_x + 150, min_y + 50),
                (min_x + 150, min_y + 150),
                (min_x + 50, min_y + 150),
            ]
        ),
        Polygon(
            [
                (min_x + 200, min_y + 200),
                (min_x + 350, min_y + 200),
                (min_x + 350, min_y + 300),
                (min_x + 200, min_y + 300),
            ]
        ),
        Polygon(
            [
                (min_x + 400, min_y + 400),
                (max_x - 20, min_y + 400),
                (max_x - 20, max_y - 20),
                (min_x + 400, max_y - 20),
            ]
        ),
    ]

    gdf = gpd.GeoDataFrame(
        {"label": ["mining", "mining", "mining"]},
        geometry=polygons,
        crs=TARGET_CRS,
    )
    gdf.to_file(labels_path, driver="GPKG")

    return labels_path


@pytest.fixture
def synthetic_labels_with_invalid(tmp_path: Path) -> Path:
    """Create a GeoPackage with one valid and one invalid geometry."""
    labels_path = tmp_path / "labels_with_invalid.gpkg"

    min_x, min_y, _, _ = _image_bounds()
    valid_polygon = Polygon(
        [
            (min_x + 50, min_y + 50),
            (min_x + 150, min_y + 50),
            (min_x + 150, min_y + 150),
            (min_x + 50, min_y + 150),
        ]
    )
    # Invalid: self-intersecting bowtie
    invalid_polygon = Polygon(
        [
            (min_x + 200, min_y + 200),
            (min_x + 300, min_y + 300),
            (min_x + 300, min_y + 200),
            (min_x + 200, min_y + 300),
        ]
    )

    gdf = gpd.GeoDataFrame(
        {"label": ["mining", "mining"]},
        geometry=[valid_polygon, invalid_polygon],
        crs=TARGET_CRS,
    )
    gdf.to_file(labels_path, driver="GPKG")

    return labels_path


@pytest.fixture
def empty_geopackage(tmp_path: Path) -> Path:
    """Create an empty GeoPackage file."""
    labels_path = tmp_path / "empty_labels.gpkg"
    gdf = gpd.GeoDataFrame(
        {"label": []},
        geometry=gpd.GeoSeries([], dtype="geometry"),
        crs=TARGET_CRS,
    )
    gdf.to_file(labels_path, driver="GPKG")
    return labels_path
