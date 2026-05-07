"""Tests for model instantiation."""

import torch

from goldmine_watch.models.unet import UNetModel, get_model


class TestModelInstantiation:
    """Test suite for model definition and forward pass."""

    def test_unet_instantiates(self) -> None:
        """Should create a U-Net model without errors."""
        model = get_model(in_channels=9)
        assert isinstance(model, UNetModel)

    def test_forward_pass_shape(self) -> None:
        """Should return a probability mask of the correct shape."""
        model = get_model(in_channels=9)
        model.eval()
        x = torch.randn(2, 9, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_model_save_load(self, tmp_path) -> None:
        """Should save and load a checkpoint preserving weights."""
        model = get_model(in_channels=9)
        x = torch.randn(1, 9, 256, 256)
        model.eval()
        with torch.no_grad():
            out_before = model(x)

        checkpoint_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), checkpoint_path)

        model_loaded = get_model(in_channels=9)
        model_loaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model_loaded.eval()
        with torch.no_grad():
            out_after = model_loaded(x)

        assert torch.allclose(out_before, out_after)
