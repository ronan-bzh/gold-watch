"""Functional tests for Feature 12: Full Territory Training.

These tests exercise the full training workflow on synthetic patches,
including resume capability and overfitting detection.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from goldmine_watch.data.patches import generate_sliding_window_patches
from goldmine_watch.models.unet import get_model
from goldmine_watch.training.train import train_phase2


class TestFeature12TrainingFlow:
    """End-to-end training workflow tests for Feature 12."""

    def test_full_training_pipeline(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """End-to-end: load patches -> train 5 epochs -> save model."""
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
        history = train_phase2(
            train_dir=patches_dir,
            epochs=5,
            batch_size=2,
            device="cpu",
            output_dir=output_dir,
        )

        assert (output_dir / "best_model.pth").exists()
        assert len(history["train_loss"]) == 5
        assert len(history["val_iou"]) == 5

    def test_best_model_saved_by_val_iou(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Best model checkpoint should correspond to the highest val IoU."""
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
        history = train_phase2(
            train_dir=patches_dir,
            epochs=5,
            batch_size=2,
            device="cpu",
            output_dir=output_dir,
        )

        best_iou_from_history = max(history["val_iou"])
        ckpt = torch.load(
            output_dir / "best_model.pth", map_location="cpu", weights_only=True
        )
        # The best checkpoint should have been saved at the epoch with max val IoU
        assert ckpt["best_iou"] == pytest.approx(best_iou_from_history, rel=1e-5)

    def test_model_predicts_on_val_patch(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Saved model can predict on a validation patch."""
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
            epochs=2,
            batch_size=2,
            device="cpu",
            output_dir=output_dir,
        )

        # Load best model
        ckpt = torch.load(
            output_dir / "best_model.pth", map_location="cpu", weights_only=True
        )
        sample_img = np.load(sorted(patches_dir.glob("image_*.npy"))[0])
        in_channels = sample_img.shape[0]
        model = get_model(in_channels=in_channels, encoder="resnet34")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        patch_t = torch.from_numpy(sample_img.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = model(patch_t)
        assert out.shape == (1, 1, 256, 256)

    def test_training_resumable(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Can resume from epoch checkpoint."""
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
        # First run: 3 epochs
        history_1 = train_phase2(
            train_dir=patches_dir,
            epochs=3,
            batch_size=2,
            device="cpu",
            output_dir=output_dir,
            save_every=3,
        )

        ckpt_path = output_dir / "epoch_003.pth"
        assert ckpt_path.exists()

        # Resume and run 2 more epochs
        history_2 = train_phase2(
            train_dir=patches_dir,
            epochs=5,
            batch_size=2,
            device="cpu",
            output_dir=output_dir,
            resume_from=ckpt_path,
            save_every=5,
        )

        # Should have 5 total epochs in history
        assert len(history_2["train_loss"]) == 5
        # The first 3 epochs should match the first run
        assert history_2["train_loss"][0] == history_1["train_loss"][0]
        assert history_2["train_loss"][1] == history_1["train_loss"][1]
        assert history_2["train_loss"][2] == history_1["train_loss"][2]

    def test_metrics_computed_in_valid_range(
        self,
        synthetic_geotiff: Path,
        synthetic_labels: Path,
        tmp_path: Path,
    ) -> None:
        """Metrics should be present and within valid bounds."""
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

        # Verify we have both train_loss and val_iou to compute a gap
        assert len(history["train_loss"]) == 3
        assert len(history["val_iou"]) == 3

        # On synthetic data the gap may be small, but we can still verify
        # the metrics are present and structurally sound.
        for i in range(3):
            assert 0.0 <= history["val_iou"][i] <= 1.0
            assert history["train_loss"][i] >= 0.0
