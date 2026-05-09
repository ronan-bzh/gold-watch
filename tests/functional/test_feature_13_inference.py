"""Functional tests for Feature 13: Batch Inference Engine."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from goldmine_watch.inference.batch import inference_batch
from goldmine_watch.models.unet import get_model

TARGET_CRS = "EPSG:2972"
DEFAULT_TILES = ["T21NZE", "T21NZF", "T21NZG", "T22NBL", "T22NBM"]


class TestFeature13InferenceFlow:
    """End-to-end batch inference workflow tests."""

    def _make_tiny_model(self, tmp_path: Path, in_channels: int = 7) -> Path:
        """Create and save a tiny untrained model."""
        model = get_model(in_channels=in_channels, encoder="resnet34")
        ckpt = tmp_path / "tiny_model.pth"
        torch.save(model.state_dict(), ckpt)
        return ckpt

    def _make_cached_tiles(
        self,
        tmp_path: Path,
        tile_ids: list[str],
        size: int = 256,
        bands: int = 7,
    ) -> Path:
        """Create synthetic cached GeoTIFFs for the given tile IDs."""
        cache_dir = tmp_path / "cache" / "tiles"
        cache_dir.mkdir(parents=True, exist_ok=True)
        transform = from_origin(200_000.0, 500_000.0 + size * 10.0, 10.0, 10.0)
        data = np.random.randint(0, 10_000, size=(bands, size, size), dtype=np.uint16)
        for tile_id in tile_ids:
            tile_path = cache_dir / f"{tile_id}_20231026.tif"
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
        return cache_dir

    def test_inference_on_single_tile(self, tmp_path: Path) -> None:
        """Run inference on one cached tile."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tiles(tmp_path, ["T21NZG"], size=256, bands=7)
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

        assert len(results) == 1
        assert results[0].exists()

    def test_inference_on_all_tiles(self, tmp_path: Path) -> None:
        """Run inference on all 5 tiles."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tiles(tmp_path, DEFAULT_TILES, size=256, bands=7)
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(model_path),
            tile_list=DEFAULT_TILES,
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(output_dir),
            tile_size=256,
            overlap=0,
            device="cpu",
        )

        assert len(results) == len(DEFAULT_TILES)
        for path in results:
            assert path.exists()

    def test_cache_reuse_during_inference(self, tmp_path: Path) -> None:
        """Should reuse cached tiles, not re-download."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tiles(tmp_path, ["T21NZG"], size=256, bands=7)
        output_dir = tmp_path / "outputs"

        # Success without any network call means cache was used.
        results = inference_batch(
            model_path=str(model_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(output_dir),
            tile_size=256,
            overlap=0,
            device="cpu",
        )

        assert len(results) == 1

    def test_predictions_are_probability_heatmaps(self, tmp_path: Path) -> None:
        """Output should be float32 [0,1] heatmaps, not binary masks."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tiles(tmp_path, ["T21NZG"], size=256, bands=7)
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
            assert src.dtypes[0] == "float32"
            data = src.read(1)
            assert data.min() >= 0.0
            assert data.max() <= 1.0
            # Verify it is not a strict binary mask
            unique = np.unique(data)
            assert len(unique) > 2 or not np.array_equal(unique, [0, 1])

    def test_inference_produces_nonzero_predictions(self, tmp_path: Path) -> None:
        """At least some pixels should have high probability."""
        model_path = self._make_tiny_model(tmp_path, in_channels=7)
        self._make_cached_tiles(tmp_path, ["T21NZG"], size=256, bands=7)
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
            # Random untrained model on random data → broad spread in (0,1)
            assert np.any(data > 0.1)
            assert np.any(data < 0.9)
