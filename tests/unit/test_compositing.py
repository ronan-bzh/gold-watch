"""Tests for temporal compositing."""

import numpy as np
import pytest
import xarray as xr

from goldmine_watch.data.stac import _composite_scenes


class TestCompositing:
    """Unit tests for the compositing helper."""

    def test_median_reduces_clouds(self) -> None:
        """Median of cloudy+clear scenes should be closer to clear."""
        # Scene 0 and 2 are clear (100), scene 1 is an outlier (1000)
        data = np.ones((3, 2, 4, 4), dtype=np.float32)
        data[0] = 100.0
        data[1] = 1000.0  # outlier / cloud
        data[2] = 100.0

        stack = xr.DataArray(
            data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": [0, 1, 2],
                "band": ["B02", "B03"],
            },
        )

        composite = _composite_scenes(stack, "median")
        expected = np.full((2, 4, 4), 100.0, dtype=np.float32)
        np.testing.assert_array_equal(composite.values, expected)

    def test_composite_same_shape_as_input(self) -> None:
        """Output shape should match a single scene (minus time dim)."""
        data = np.random.rand(5, 3, 64, 64).astype(np.float32)
        stack = xr.DataArray(
            data,
            dims=["time", "band", "y", "x"],
            coords={"time": range(5), "band": ["B02", "B03", "B04"]},
        )
        composite = _composite_scenes(stack, "median")
        assert composite.shape == (3, 64, 64)

    def test_invalid_aggregator_raises(self) -> None:
        """aggregator='max' should raise ValueError."""
        stack = xr.DataArray(
            np.ones((2, 1, 4, 4), dtype=np.float32),
            dims=["time", "band", "y", "x"],
        )
        with pytest.raises(ValueError, match="aggregator must be 'median' or 'mean'"):
            _composite_scenes(stack, "max")
