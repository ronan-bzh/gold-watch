"""Functional tests for Feature 5: Single-Scene Inference & Evaluation.

These tests exercise the full inference pipeline on a large synthetic image,
including tiling, blending, and saving a probability raster.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from goldmine_watch.inference.predict_big import predict_big_image
from goldmine_watch.inference.tiler import tile_image
from goldmine_watch.models.unet import get_model

TARGET_CRS = "EPSG:2972"


class TestFeature5InferenceFlow:
    """End-to-end inference workflow tests."""

    def _train_tiny_model(self, tmp_path: Path, in_channels: int = 7) -> Path:
        """Create and save a tiny untrained model for inference tests."""
        model = get_model(in_channels=in_channels, encoder="resnet18")
        ckpt = tmp_path / "tiny_model.pth"
        torch.save(model.state_dict(), ckpt)
        return ckpt

    def _make_big_image(self, tmp_path: Path, size: int = 512, bands: int = 7) -> Path:
        """Create a synthetic large GeoTIFF."""
        image_path = tmp_path / "big_image.tif"
        transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(bands, size, size), dtype=np.uint16)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=bands,
            dtype=data.dtype,
            crs=TARGET_CRS,
            transform=transform,
        ) as dst:
            dst.write(data)
        return image_path

    def test_full_inference_produces_geotiff(
        self, tmp_path: Path
    ) -> None:
        """Running inference on a big image produces a valid GeoTIFF."""
        image_path = self._make_big_image(tmp_path, size=512, bands=7)
        model_path = self._train_tiny_model(tmp_path, in_channels=7)
        output_path = tmp_path / "prediction.tif"

        result = predict_big_image(
            image_path,
            model_path,
            output_path,
            tile_size=256,
            overlap=64,
            in_channels=7,
            device="cpu",
        )

        assert result.exists()
        with rasterio.open(result) as src:
            assert src.count == 1
            assert src.dtypes[0] == "float32"
            assert src.width == 512
            assert src.height == 512
            assert src.crs is not None

    def test_tiling_creates_expected_number_of_tiles(
        self, tmp_path: Path
    ) -> None:
        """A 512x512 image with 256 tiles and 64 overlap yields 9 tiles."""
        image_path = self._make_big_image(tmp_path, size=512, bands=7)
        tiles = tile_image(image_path, tile_size=256, overlap=64)
        # stride = 192; 512/192 -> 3 in each direction -> 9 tiles
        assert len(tiles) == 9
        for tile, window in tiles:
            assert tile.shape[1] == 256
            assert tile.shape[2] == 256

    def test_inference_on_small_image(
        self, tmp_path: Path
    ) -> None:
        """A 256x256 image with tile_size=256 produces exactly one tile."""
        image_path = self._make_big_image(tmp_path, size=256, bands=7)
        model_path = self._train_tiny_model(tmp_path, in_channels=7)
        output_path = tmp_path / "prediction_small.tif"

        result = predict_big_image(
            image_path,
            model_path,
            output_path,
            tile_size=256,
            overlap=0,
            in_channels=7,
            device="cpu",
        )

        assert result.exists()
        with rasterio.open(result) as src:
            assert src.width == 256
            assert src.height == 256

    def test_probability_range_0_to_1(
        self, tmp_path: Path
    ) -> None:
        """Output raster values are in [0, 1]."""
        image_path = self._make_big_image(tmp_path, size=256, bands=7)
        model_path = self._train_tiny_model(tmp_path, in_channels=7)
        output_path = tmp_path / "prediction_probs.tif"

        predict_big_image(
            image_path,
            model_path,
            output_path,
            tile_size=256,
            overlap=0,
            in_channels=7,
            device="cpu",
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert data.min() >= 0.0
            assert data.max() <= 1.0

    def test_different_tile_sizes_produce_same_spatial_extent(
        self, tmp_path: Path
    ) -> None:
        """tile_size=128 and tile_size=256 both produce 512x512 outputs."""
        image_path = self._make_big_image(tmp_path, size=512, bands=7)
        model_path = self._train_tiny_model(tmp_path, in_channels=7)

        out_256 = tmp_path / "out_256.tif"
        out_128 = tmp_path / "out_128.tif"

        predict_big_image(
            image_path, model_path, out_256,
            tile_size=256, overlap=64, in_channels=7, device="cpu",
        )
        predict_big_image(
            image_path, model_path, out_128,
            tile_size=128, overlap=32, in_channels=7, device="cpu",
        )

        with rasterio.open(out_256) as s1, rasterio.open(out_128) as s2:
            assert s1.width == s2.width == 512
            assert s1.height == s2.height == 512
            assert s1.crs == s2.crs
