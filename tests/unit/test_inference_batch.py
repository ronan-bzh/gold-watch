"""Unit tests for Feature 13: Batch Inference Engine."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from goldmine_watch.data.tile_registry import TileRegistry
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
        """Create a synthetic cached GeoTIFF for *tile_id* inside French Guiana."""
        cache_dir = tmp_path / "cache" / "tiles"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tile_path = cache_dir / f"{tile_id}_20231026.tif"
        # WGS84 bounds inside French Guiana (~4.0°N, -53.0°W)
        transform = from_origin(-53.05, 4.05, 0.0001, 0.0001)
        data = np.random.randint(0, 10_000, size=(bands, size, size), dtype=np.uint16)
        with rasterio.open(
            tile_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=bands,
            dtype=data.dtype,
            crs="EPSG:4326",
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

    def _register_tile(self, tmp_path: Path, tile_id: str) -> None:
        """Register the synthetic tile in the SQLite registry."""
        db_path = tmp_path / "tiles.db"
        schema_path = tmp_path / "schema.sql"
        schema_path.write_text(
            """
            CREATE TABLE IF NOT EXISTS tiles (
                id INTEGER PRIMARY KEY,
                tile_id TEXT NOT NULL,
                date TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                west REAL NOT NULL,
                south REAL NOT NULL,
                east REAL NOT NULL,
                north REAL NOT NULL,
                crs TEXT NOT NULL DEFAULT 'EPSG:4326',
                width INTEGER,
                height INTEGER,
                bands INTEGER,
                size_bytes INTEGER,
                source TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(tile_id, date)
            );
            CREATE TABLE IF NOT EXISTS fg_boundary (
                id INTEGER PRIMARY KEY,
                name TEXT,
                west REAL NOT NULL,
                south REAL NOT NULL,
                east REAL NOT NULL,
                north REAL NOT NULL
            );
            INSERT OR IGNORE INTO fg_boundary (id, name, west, south, east, north)
            VALUES (1, 'french_guiana', -54.6, 2.1, -51.6, 5.8);
            """
        )
        reg = TileRegistry(db_path=str(db_path), schema_path=str(schema_path))
        tile_path = tmp_path / "cache" / "tiles" / f"{tile_id}_20231026.tif"
        with rasterio.open(tile_path) as src:
            bounds = src.bounds
        reg.register_tile(
            tile_id=tile_id,
            date="20231026",
            filepath=str(tile_path),
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            crs="EPSG:4326",
            width=256,
            height=256,
            bands=7,
            size_bytes=tile_path.stat().st_size,
            source="test",
        )

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
        self._register_tile(tmp_path, "T21NZG")
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(ckpt_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            db_path=str(tmp_path / "tiles.db"),
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
        self._register_tile(tmp_path, "T21NZG")
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(model_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            db_path=str(tmp_path / "tiles.db"),
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
        self._register_tile(tmp_path, "T21NZG")
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(model_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            db_path=str(tmp_path / "tiles.db"),
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
        self._register_tile(tmp_path, "T21NZG")
        output_dir = tmp_path / "outputs"

        results = inference_batch(
            model_path=str(model_path),
            tile_list=["T21NZG"],
            cache_dir=str(tmp_path / "cache"),
            db_path=str(tmp_path / "tiles.db"),
            output_dir=str(output_dir),
            tile_size=256,
            overlap=0,
            device="cpu",
        )

        with rasterio.open(results[0]) as src:
            data = src.read(1)
            assert data.min() >= 0.0
            assert data.max() <= 1.0
