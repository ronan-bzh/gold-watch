"""Unit tests for Feature 12: Full Territory Training."""

from pathlib import Path

import pytest
import torch

from goldmine_watch.data.patches import generate_sliding_window_patches
from goldmine_watch.models.unet import get_model
from goldmine_watch.training.train import train_phase2


class TestPhase2Training:
    """Unit tests for the phase-2 training function."""

    def test_model_loads_with_correct_channels(self) -> None:
        """Model should accept 7 input channels (Sentinel-2 bands)."""
        model = get_model(in_channels=7, encoder="resnet34")
        dummy = torch.randn(1, 7, 256, 256)
        out = model(dummy)
        assert out.shape == (1, 1, 256, 256)

    def test_pos_weight_computed(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """pos_weight should be > 1.0 for imbalanced data."""
        patches_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=patches_dir,
        )

        train_phase2(
            train_dir=patches_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            output_dir=tmp_path / "models",
        )

        captured = capsys.readouterr()
        assert "pos_weight:" in captured.out
        # Extract pos_weight value
        line = [ln for ln in captured.out.split("\n") if "pos_weight:" in ln][0]
        pos_weight = float(line.split("pos_weight:")[1].strip())
        assert pos_weight > 1.0

    def test_training_loss_stable_or_decreases(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Loss should not increase by more than 50% over first few epochs."""
        patches_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=patches_dir,
        )

        history = train_phase2(
            train_dir=patches_dir,
            epochs=3,
            batch_size=2,
            device="cpu",
            output_dir=tmp_path / "models",
        )

        assert history["train_loss"][-1] < history["train_loss"][0] * 1.5

    def test_checkpoint_saved(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Should save best_model.pth and epoch checkpoints."""
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
        train_phase2(
            train_dir=patches_dir,
            epochs=5,
            batch_size=2,
            device="cpu",
            output_dir=output_dir,
            save_every=5,
        )

        assert (output_dir / "best_model.pth").exists()
        assert (output_dir / "epoch_005.pth").exists()

        # Verify checkpoint contains expected keys
        ckpt = torch.load(output_dir / "epoch_005.pth", map_location="cpu", weights_only=True)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "history" in ckpt

    def test_history_has_expected_keys(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """History should contain train_loss, val_loss, val_iou, val_f1."""
        patches_dir = tmp_path / "patches"
        generate_sliding_window_patches(
            synthetic_geotiff,
            synthetic_labels,
            patch_size=256,
            stride=256,
            max_patches=500,
            output_dir=patches_dir,
        )

        history = train_phase2(
            train_dir=patches_dir,
            epochs=2,
            batch_size=2,
            device="cpu",
            output_dir=tmp_path / "models",
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_iou" in history
        assert "val_f1" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_iou"]) == 2


class TestModelCapacity:
    """Tests comparing model capacities."""

    def test_resnet34_has_more_params_than_resnet18(self) -> None:
        """ResNet-34 should be larger than ResNet-18."""
        model_18 = get_model(in_channels=7, encoder="resnet18")
        model_34 = get_model(in_channels=7, encoder="resnet34")

        params_18 = sum(p.numel() for p in model_18.parameters())
        params_34 = sum(p.numel() for p in model_34.parameters())

        assert params_34 > params_18
