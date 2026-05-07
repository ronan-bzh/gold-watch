"""Tests for inference modules."""

from pathlib import Path

import numpy as np
import rasterio
import torch

from goldmine_watch.inference.blender import blend_predictions, normalize_canvas
from goldmine_watch.inference.postprocess import postprocess
from goldmine_watch.inference.predict import predict_patch, save_prediction_visual
from goldmine_watch.inference.tiler import tile_image
from goldmine_watch.models.unet import get_model


class TestTiler:
    """Test sliding window tiling."""

    def test_tile_image(self, synthetic_geotiff: Path) -> None:
        """Should split image into tiles."""
        tiles = tile_image(synthetic_geotiff, tile_size=128, overlap=0)
        assert len(tiles) > 0
        for tile, _window in tiles:
            assert tile.shape[1:] == (128, 128)


class TestBlender:
    """Test blending of overlapping predictions."""

    def test_blend_and_normalize(self) -> None:
        """Should average overlapping regions correctly."""
        canvas = np.zeros((256, 256), dtype=np.float32)
        counts = np.zeros((256, 256), dtype=np.float32)
        pred = np.ones((128, 128), dtype=np.float32)

        blend_predictions(canvas, counts, pred, x=0, y=0, tile_size=128)
        blend_predictions(canvas, counts, pred, x=64, y=0, tile_size=128)

        result = normalize_canvas(canvas, counts)
        # Overlap region (x=64..128) should be averaged to 1.0
        assert np.allclose(result[0:128, 0:128], 1.0)


class TestPredictPatch:
    """Test single-patch prediction."""

    def test_predict_shape(self, tmp_path: Path) -> None:
        """Should return a probability map of the same spatial size."""
        model = get_model(in_channels=9)
        ckpt = tmp_path / "model.pth"
        torch.save(model.state_dict(), ckpt)

        image_patch = np.random.randn(9, 256, 256).astype(np.float32)
        pred = predict_patch(image_patch, ckpt, in_channels=9, device="cpu")
        assert pred.shape == (256, 256)
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_save_visual(self, tmp_path: Path) -> None:
        """Should save a PNG file."""
        image_patch = np.random.randint(0, 10_000, size=(9, 256, 256), dtype=np.uint16)
        pred = np.random.rand(256, 256).astype(np.float32)
        out_path = save_prediction_visual(image_patch, pred, tmp_path / "pred.png")
        assert out_path.exists()


class TestPostprocess:
    """Test raster-to-vector post-processing."""

    def test_postprocess_creates_geopackage(self, tmp_path: Path) -> None:
        """Should create a GeoPackage from a probability raster."""
        raster_path = tmp_path / "probs.tif"
        output_path = tmp_path / "polygons.gpkg"

        # Create a simple probability raster with one bright square
        probs = np.zeros((256, 256), dtype=np.float32)
        probs[50:150, 50:150] = 0.8

        transform = rasterio.Affine.identity() * rasterio.Affine.translation(0, 0)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=1,
            dtype=probs.dtype,
            crs="EPSG:2972",
            transform=transform,
        ) as dst:
            dst.write(probs, 1)

        result = postprocess(raster_path, output_path, threshold=0.5, min_area_pixels=1)
        assert result.exists()
