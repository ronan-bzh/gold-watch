"""Smoke tests for training script."""

from pathlib import Path

import pytest

from goldmine_watch.training.train import train_fake


class TestTrainingSmoke:
    """Quick smoke tests for the training loop."""

    @pytest.mark.slow
    def test_train_fake_runs(
        self, synthetic_geotiff: Path, synthetic_labels: Path, tmp_path: Path
    ) -> None:
        """Should run training on fake data without crashing and produce a checkpoint."""
        # Train for just 2 epochs
        train_fake(
            synthetic_geotiff, synthetic_labels, epochs=2, device="cpu", output_dir=tmp_path
        )

        assert (tmp_path / "milestone3_fake.pth").exists()
