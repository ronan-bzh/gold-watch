"""Unit tests for the build_training_dataset module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from goldmine_watch.data.build_training_dataset import (
    _extract_background_patches,
    _extract_mine_centered_patches,
    _spatial_split_tiles,
    build_training_dataset,
)


def _make_two_tile_clusters(mines_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Return a fake 2-tile cluster dict for unit tests."""
    gdf = gpd.read_file(mines_path)
    mid = len(gdf) // 2
    return {
        "T21NZG": gdf.iloc[:mid].reset_index(drop=True),
        "T22NBM": gdf.iloc[mid:].reset_index(drop=True),
    }


if TYPE_CHECKING:
    pass


def _make_synthetic_tile(
    path: Path,
    width: int = 512,
    height: int = 512,
    count: int = 7,
    crs: str = "EPSG:32622",
) -> Path:
    """Create a synthetic GeoTIFF tile for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(200_000.0, 500_000.0, 10.0, 10.0)
    data = np.random.randint(0, 1000, size=(count, height, width), dtype=np.uint16)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


def _make_mines_geojson(path: Path, crs: str = "EPSG:32622") -> Path:
    """Create a GeoJSON with a few synthetic mine polygons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mines centered at various positions in the tile
    polygons = [
        Polygon(
            [
                (200_050, 499_950),
                (200_150, 499_950),
                (200_150, 500_050),
                (200_050, 500_050),
            ]
        ),
        Polygon(
            [
                (200_300, 499_700),
                (200_400, 499_700),
                (200_400, 499_800),
                (200_300, 499_800),
            ]
        ),
    ]
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=polygons,
        crs=crs,
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


class TestBuildTrainingDataset:
    """Tests for the full dataset build pipeline."""

    def test_output_dirs_created(self, tmp_path: Path) -> None:
        """Should create train/ and val/ directories."""
        mines_path = _make_mines_geojson(tmp_path / "mines.geojson")
        tile_a = _make_synthetic_tile(tmp_path / "cache" / "T21NZG_20231026.tif")
        tile_b = _make_synthetic_tile(tmp_path / "cache" / "T22NBM_20231026.tif")
        tile_map = {"T21NZG": tile_a, "T22NBM": tile_b}

        output_dir = tmp_path / "splits"

        with (
            patch("goldmine_watch.data.build_training_dataset.TileCache") as mock_cache_cls,
            patch(
                "goldmine_watch.data.build_training_dataset.cluster_mines_by_tile",
                return_value=_make_two_tile_clusters(mines_path),
            ),
        ):
            mock_cache = MagicMock()
            mock_cache.get_tile.side_effect = lambda tile_id, **_: tile_map[tile_id]
            mock_cache_cls.return_value = mock_cache

            build_training_dataset(
                mines_geojson=str(mines_path),
                output_dir=str(output_dir),
                num_background_per_tile=10,
                patch_size=256,
                train_val_ratio=(0.5, 0.5),
                random_seed=42,
                cache_dir=str(tmp_path / "cache"),
            )

        assert (output_dir / "train").exists()
        assert (output_dir / "train").is_dir()
        assert (output_dir / "val").exists()
        assert (output_dir / "val").is_dir()

    def test_patches_are_numpy_arrays(self, tmp_path: Path) -> None:
        """Saved patches should be valid .npy files."""
        mines_path = _make_mines_geojson(tmp_path / "mines.geojson")
        tile_a = _make_synthetic_tile(tmp_path / "cache" / "T21NZG_20231026.tif")
        tile_b = _make_synthetic_tile(tmp_path / "cache" / "T22NBM_20231026.tif")
        tile_map = {"T21NZG": tile_a, "T22NBM": tile_b}
        output_dir = tmp_path / "splits"

        with (
            patch("goldmine_watch.data.build_training_dataset.TileCache") as mock_cache_cls,
            patch(
                "goldmine_watch.data.build_training_dataset.cluster_mines_by_tile",
                return_value=_make_two_tile_clusters(mines_path),
            ),
        ):
            mock_cache = MagicMock()
            mock_cache.get_tile.side_effect = lambda tile_id, **_: tile_map[tile_id]
            mock_cache_cls.return_value = mock_cache

            build_training_dataset(
                mines_geojson=str(mines_path),
                output_dir=str(output_dir),
                num_background_per_tile=5,
                patch_size=256,
                train_val_ratio=(0.5, 0.5),
                random_seed=42,
                cache_dir=str(tmp_path / "cache"),
            )

        all_image_files = []
        all_mask_files = []
        for split in ["train", "val"]:
            image_files = list((output_dir / split).glob("image_*.npy"))
            mask_files = list((output_dir / split).glob("mask_*.npy"))
            all_image_files.extend(image_files)
            all_mask_files.extend(mask_files)

        assert len(all_image_files) > 0, "No image patches found"
        assert len(all_mask_files) > 0, "No mask patches found"

        for img_path in all_image_files:
            arr = np.load(img_path)
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 3  # (bands, height, width)

        for msk_path in all_mask_files:
            arr = np.load(msk_path)
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 2  # (height, width)

    def test_train_val_split_ratio(self, tmp_path: Path) -> None:
        """Should produce approximately the requested train/val ratio."""
        mines_path = _make_mines_geojson(tmp_path / "mines.geojson")
        tile_a = _make_synthetic_tile(tmp_path / "cache" / "T21NZG_20231026.tif")
        tile_b = _make_synthetic_tile(tmp_path / "cache" / "T22NBM_20231026.tif")
        tile_map = {"T21NZG": tile_a, "T22NBM": tile_b}
        output_dir = tmp_path / "splits"

        with (
            patch("goldmine_watch.data.build_training_dataset.TileCache") as mock_cache_cls,
            patch(
                "goldmine_watch.data.build_training_dataset.cluster_mines_by_tile",
                return_value=_make_two_tile_clusters(mines_path),
            ),
        ):
            mock_cache = MagicMock()
            mock_cache.get_tile.side_effect = lambda tile_id, **_: tile_map[tile_id]
            mock_cache_cls.return_value = mock_cache

            stats = build_training_dataset(
                mines_geojson=str(mines_path),
                output_dir=str(output_dir),
                num_background_per_tile=10,
                patch_size=256,
                train_val_ratio=(0.8, 0.2),
                random_seed=42,
                cache_dir=str(tmp_path / "cache"),
            )

        total = stats["total_patches"]
        # With two tiles at 80/20, one tile should go to train and one to val
        assert stats["train_patches"] + stats["val_patches"] == total
        assert stats["train_patches"] > 0
        assert stats["val_patches"] > 0

    def test_some_positive_patches_exist(self, tmp_path: Path) -> None:
        """At least some patches should contain mining pixels."""
        mines_path = _make_mines_geojson(tmp_path / "mines.geojson")
        tile_a = _make_synthetic_tile(tmp_path / "cache" / "T21NZG_20231026.tif")
        tile_b = _make_synthetic_tile(tmp_path / "cache" / "T22NBM_20231026.tif")
        tile_map = {"T21NZG": tile_a, "T22NBM": tile_b}
        output_dir = tmp_path / "splits"

        with (
            patch("goldmine_watch.data.build_training_dataset.TileCache") as mock_cache_cls,
            patch(
                "goldmine_watch.data.build_training_dataset.cluster_mines_by_tile",
                return_value=_make_two_tile_clusters(mines_path),
            ),
        ):
            mock_cache = MagicMock()
            mock_cache.get_tile.side_effect = lambda tile_id, **_: tile_map[tile_id]
            mock_cache_cls.return_value = mock_cache

            stats = build_training_dataset(
                mines_geojson=str(mines_path),
                output_dir=str(output_dir),
                num_background_per_tile=5,
                patch_size=256,
                train_val_ratio=(0.5, 0.5),
                random_seed=42,
                cache_dir=str(tmp_path / "cache"),
            )

        assert stats["positive_patches"] > 0


class TestMineCenteredExtraction:
    """Tests for mine-centered patch extraction."""

    def test_mine_in_patch(self, tmp_path: Path) -> None:
        """Mine-centered patch should contain the target mine."""
        tile_path = _make_synthetic_tile(tmp_path / "tile.tif")

        # Create mines in the tile's CRS (EPSG:32622)
        polygons = [
            Polygon(
                [
                    (200_100, 499_900),
                    (200_200, 499_900),
                    (200_200, 500_000),
                    (200_100, 500_000),
                ]
            ),
        ]
        mines_gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=polygons,
            crs="EPSG:32622",
        )

        # Burn mask
        from goldmine_watch.data.ingest import burn_mask

        mask = burn_mask(mines_gdf, tile_path)
        patches = _extract_mine_centered_patches(tile_path, mask, mines_gdf, patch_size=256)

        assert len(patches) == 1
        image_patch, mask_patch = patches[0]
        assert np.any(mask_patch > 0), "Patch should contain mine pixels"

    def test_patch_size_correct(self, tmp_path: Path) -> None:
        """All patches should be patch_size x patch_size."""
        tile_path = _make_synthetic_tile(tmp_path / "tile.tif", width=1024, height=1024)

        polygons = [
            Polygon(
                [
                    (200_100, 499_900),
                    (200_200, 499_900),
                    (200_200, 500_000),
                    (200_100, 500_000),
                ]
            ),
            Polygon(
                [
                    (200_400, 499_600),
                    (200_500, 499_600),
                    (200_500, 499_700),
                    (200_400, 499_700),
                ]
            ),
        ]
        mines_gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=polygons,
            crs="EPSG:32622",
        )

        from goldmine_watch.data.ingest import burn_mask

        mask = burn_mask(mines_gdf, tile_path)
        patches = _extract_mine_centered_patches(tile_path, mask, mines_gdf, patch_size=128)

        assert len(patches) == 2
        for image_patch, mask_patch in patches:
            assert image_patch.shape[1:] == (128, 128)
            assert mask_patch.shape == (128, 128)


class TestSpatialSplitTiles:
    """Tests for spatial train/val tile splitting."""

    def test_splits_are_disjoint(self) -> None:
        """Train and val tile sets should not overlap."""
        train, val = _spatial_split_tiles(
            ["A", "B", "C", "D", "E"],
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
        )
        assert set(train).isdisjoint(set(val))

    def test_all_tiles_assigned(self) -> None:
        """Every tile should be in exactly one split."""
        train, val = _spatial_split_tiles(
            ["A", "B", "C"],
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
        )
        assert set(train) | set(val) == {"A", "B", "C"}

    def test_at_least_one_per_split(self) -> None:
        """Each split should have at least one tile."""
        train, val = _spatial_split_tiles(
            ["A", "B"],
            train_val_ratio=(0.8, 0.2),
            random_seed=42,
        )
        assert len(train) >= 1
        assert len(val) >= 1

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce the same split."""
        train1, val1 = _spatial_split_tiles(
            ["A", "B", "C", "D", "E"],
            train_val_ratio=(0.8, 0.2),
            random_seed=123,
        )
        train2, val2 = _spatial_split_tiles(
            ["A", "B", "C", "D", "E"],
            train_val_ratio=(0.8, 0.2),
            random_seed=123,
        )
        assert train1 == train2
        assert val1 == val2

    def test_fewer_than_two_tiles_raises(self) -> None:
        """Spatial split with <2 tiles should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 tiles"):
            _spatial_split_tiles(
                ["A"],
                train_val_ratio=(0.8, 0.2),
                random_seed=42,
            )


class TestBackgroundExtraction:
    """Tests for background patch extraction."""

    def test_background_has_no_mines(self, tmp_path: Path) -> None:
        """Background patches should contain no mine pixels."""
        tile_path = _make_synthetic_tile(tmp_path / "tile.tif", width=512, height=512)

        polygons = [
            Polygon(
                [
                    (200_100, 499_900),
                    (200_150, 499_900),
                    (200_150, 499_950),
                    (200_100, 499_950),
                ]
            ),
        ]
        mines_gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=polygons,
            crs="EPSG:32622",
        )

        from goldmine_watch.data.ingest import burn_mask

        mask = burn_mask(mines_gdf, tile_path)
        patches = _extract_background_patches(
            tile_path, mask, patch_size=64, num_patches=5, random_seed=42
        )

        assert len(patches) == 5
        for _, mask_patch in patches:
            assert not np.any(mask_patch > 0), "Background patch contains mine pixels"

    def test_respects_num_patches(self, tmp_path: Path) -> None:
        """Should return the requested number of patches when possible."""
        tile_path = _make_synthetic_tile(tmp_path / "tile.tif", width=512, height=512)
        mask = np.zeros((512, 512), dtype=np.uint8)

        patches = _extract_background_patches(
            tile_path, mask, patch_size=64, num_patches=10, random_seed=42
        )

        assert len(patches) == 10
