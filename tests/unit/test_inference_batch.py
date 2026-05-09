"""Unit tests for Feature 13: Batch Inference Engine."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from goldmine_watch.inference.batch import inference_batch
from goldmine_watch.models.unet import get_model

TARGET_CRS = "EPSG:2972"


class TestInferenceBatch:
    """Unit tests for batch inference."""

    def _make_tiny_model(self, tmp_path: Path, in_channels: int = 7) -> Path:
        """Create and save a tiny untrained model for inference tests."""
        model = get_model(in_channels=in_channels, encoder="resnet34")
        ckpt = tmp_path / "tiny_model.pth"
        torch.save(model.state_dict(), ckpt)
        return ckpt

    def _make_cached_tile(
        self,
        tmp_path: Path,
        tile_id: str,
        size: int = 512,
        bands: int = 7,
    ) -> Path:
        """Create a synthetic cached GeoTIFF for *tile_id*."""
        cache_dir = tmp_path / "cache" / "tiles"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tile_path = cache_dir / f"{tile_id}_20231026.tif"
        transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(bands, size, size), dtype=np.uint16)
        with rasterio.open(
            tile_path,
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
        return tile_path

    def test_model_loads_successfully(self, tmp_path: Path) -> None:
        """Should load a raw state_dict checkpoint without error."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        model = get_model(in_channels=7, encoder="resnet34")
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        assert model is not None

    def test_model_loads_wrapped_checkpoint(self, tmp_path: Path) -> None:
        """Should load a wrapped checkpoint (model_state_dict key) without error."""
        model = get_model(in_channels=7, encoder="resnet34")
        ckpt_path = tmp_path / "wrapped_model.pth"
        torch.save(
            {
                "epoch": 10,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
            },
            ckpt_path,
        )
        self._make_cached_tile(tmp_path, "T21NZG", size=256, bands=7)
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(ckpt_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(output_dir),
            tile_size=256,
            overlap=0,
            device="cpu",
        )

        assert len(results) == 1
        assert results[0].exists()

    def test_prediction_raster_created(self, tmp_path: Path) -> None:
        """Should create a GeoTIFF prediction file."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tile(tmp_path, "T21NZG", size=512, bands=7)
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(model_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(output_dir),
            tile_size=256,
            overlap=64,
            device="cpu",
        )

        assert len(results) == 1
        assert results[0].exists()
        assert results[0].suffix == ".tif"

    def test_prediction_has_correct_shape(self, tmp_path: Path) -> None:
        """Prediction should match input image dimensions."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tile(tmp_path, "T21NZG", size=512, bands=7)
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(model_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(output_dir),
            tile_size=256,
            overlap=64,
            device="cpu",
        )

        with rasterio.open(results[0]) as src:
            assert src.width == 512
            assert src.height == 512

    def test_probability_range(self, tmp_path: Path) -> None:
        """All values should be between 0.0 and 1.0."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tile(tmp_path, "T21NZG", size=256, bands=7)
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(model_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(output_dir),
            tile_size=256,
            overlap=0,
            device="cpu",
        )

        with rasterio.open(results[0]) as src:
            data = src.read(1)
            assert data.min() >= 0.0
            assert data.max() <= 1.0
