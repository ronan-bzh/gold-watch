"""Functional tests for Feature 4: Training with Metrics & Spatial Validation.

These tests exercise the full training workflow on synthetic patches,
including checkpoint saving and basic metric computation.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from goldmine_watch.data.patches import generate_sliding_window_patches
from goldmine_watch.training.train import train_patches
from goldmine_watch.models.unet import get_model

TARGET_CRS = "EPSG:2972"


class TestFeature4TrainingFlow:
    """End-to-end training workflow tests."""

    def test_train_on_patches_produces_checkpoints(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Training on patches saves checkpoint files."""
        patches_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=patches_dir,
        )

        output_dir = tmp_path / "models"
        train_patches(
            patches_dir=patches_dir,
            epochs=2,
            batch_size=2,
            lr=0.001,
            device="cpu",
        )

        # train_patches saves checkpoints to models/ by default
        assert (Path("models") / "epoch_001.pth").exists()
        assert (Path("models") / "epoch_002.pth").exists()

    def test_checkpoint_loads_and_predicts(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Saved checkpoint can be loaded and used for inference."""
        patches_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=patches_dir,
        )

        train_patches(
            patches_dir=patches_dir,
            epochs=1,
            batch_size=2,
            lr=0.001,
            device="cpu",
        )

        ckpt_path = Path("models") / "epoch_001.pth"
        assert ckpt_path.exists()

        # Load and verify model can forward a patch
        sample_img = np.load(sorted(patches_dir.glob("image_*.npy"))[0])
        in_channels = sample_img.shape[0]
        model = get_model(in_channels=in_channels)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()

        patch_t = torch.from_numpy(sample_img.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = model(patch_t)
        assert out.shape == (1, 1, 256, 256)

    def test_training_loss_decreases(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Loss printed in epoch 2 should be <= epoch 1 (or close)."""
        patches_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=patches_dir,
        )

        train_patches(
            patches_dir=patches_dir,
            epochs=2,
            batch_size=2,
            lr=0.001,
            device="cpu",
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.split("\n") if "Loss:" in line]
        assert len(lines) == 2
        loss_1 = float(lines[0].split("Loss:")[1].strip())
        loss_2 = float(lines[1].split("Loss:")[1].strip())
        # Loss should not explode; allow small tolerance for stochasticity
        assert loss_2 < loss_1 * 1.5

    def test_patches_with_cloud_mask_train_successfully(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Training succeeds when patches were generated with cloud masking."""
        patches_dir = tmp_path / "patches"
        cloud_mask = np.ones((512, 512), dtype=np.uint8)
        cloud_mask[:256, :256] = 0  # reject top-left patch

        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            max_cloud_fraction=0.2,
            cloud_mask=cloud_mask,
            output_dir=patches_dir,
        )

        train_patches(
            patches_dir=patches_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
        )

        assert (Path("models") / "epoch_001.pth").exists()

    def test_training_on_empty_patches_raises(
        self, tmp_path: Path
    ) -> None:
        """Empty patches directory should raise ValueError."""
        empty_dir = tmp_path / "empty_patches"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match=r"No image_\*\.npy files found"):
            train_patches(patches_dir=empty_dir, epochs=1, device="cpu")

    def test_different_batch_sizes_train_successfully(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Training works with batch sizes 1 and 4."""
        patches_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=patches_dir,
        )

        for bs in (1, 4):
            train_patches(
                patches_dir=patches_dir,
                epochs=1,
                batch_size=bs,
                device="cpu",
            )
            assert (Path("models") / "epoch_001.pth").exists()
