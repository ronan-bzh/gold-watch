"""Tests for data ingestion."""

import pytest


class TestLoadLabels:
    """Test suite for label loading and validation."""

    def test_load_valid_geopackage(self):
        """Should load a valid GeoPackage file and return a GeoDataFrame."""
        pass

    def test_skip_invalid_geometries(self):
        """Should skip invalid geometries with a warning instead of crashing."""
        pass

    def test_reproject_to_target_crs(self):
        """Should reproject labels to the pipeline-wide target CRS."""
        pass

    def test_empty_file_raises(self):
        """Should raise ValueError when the label file is empty."""
        pass
