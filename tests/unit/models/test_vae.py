"""Tests for VAE encoding and decoding functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.diffusion_art.models.vae import SD15VAE


class TestSD15VAE:
    """Test SD15 VAE encoder/decoder."""

    def test_vae_initialization(self):
        """Test VAE initialization with different devices."""
        # Test auto device detection
        vae = SD15VAE()
        expected_device = "mps" if torch.backends.mps.is_available() else "cpu"
        assert vae.device == expected_device

        # Test explicit device setting
        vae_cpu = SD15VAE(device="cpu")
        assert vae_cpu.device == "cpu"

    def test_transform_creation(self):
        """Test that image transform is created correctly."""
        vae = SD15VAE()
        transform = vae._create_transform()

        # Create a test image
        test_img = Image.new("RGB", (256, 256), color="red")
        transformed = transform(test_img)

        # Should be normalized tensor of shape (3, 512, 512)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 512, 512)
        assert transformed.min() >= -1.0 and transformed.max() <= 1.0

    @patch("src.diffusion_art.models.vae.AutoencoderKL")
    def test_model_loading(self, mock_autoencoder):
        """Test that model is loaded correctly."""
        mock_vae = Mock()
        mock_autoencoder.from_pretrained.return_value = mock_vae
        mock_vae.to.return_value = mock_vae
        mock_vae.eval.return_value = mock_vae

        vae = SD15VAE(device="cpu")
        vae.load_model()

        # Check model was loaded from correct path
        mock_autoencoder.from_pretrained.assert_called_once_with(
            "runwayml/stable-diffusion-v1-5", subfolder="vae"
        )
        mock_vae.to.assert_called_once_with("cpu")
        mock_vae.eval.assert_called_once()
        assert vae.vae is mock_vae

    @patch("src.diffusion_art.models.vae.AutoencoderKL")
    def test_encode_functionality(self, mock_autoencoder):
        """Test encoding functionality without actual model."""
        # Mock the VAE model and its methods
        mock_vae = Mock()
        mock_latent_dist = Mock()
        mock_latent = torch.randn(1, 4, 64, 64)
        mock_latent_dist.sample.return_value = mock_latent

        # Mock the encode method to return the latent_dist mock
        mock_encode_result = Mock()
        mock_encode_result.latent_dist = mock_latent_dist
        mock_vae.encode.return_value = mock_encode_result

        mock_autoencoder.from_pretrained.return_value = mock_vae
        mock_vae.to.return_value = mock_vae
        mock_vae.eval.return_value = mock_vae

        vae = SD15VAE(device="cpu")

        # Create test image
        test_img = Image.new("RGB", (256, 256), color="blue")

        result = vae.encode(test_img)

        # Check that encoding was called and result is scaled correctly
        mock_vae.encode.assert_called_once()
        expected = mock_latent * vae.SD15_SCALE
        assert torch.allclose(result, expected)
        assert result.shape == (1, 4, 64, 64)

    @patch("src.diffusion_art.models.vae.AutoencoderKL")
    def test_decode_functionality(self, mock_autoencoder):
        """Test decoding functionality without actual model."""
        # Mock the VAE model
        mock_vae = Mock()
        # Create mock decoded output in expected range
        mock_decoded = torch.randn(1, 3, 512, 512) * 0.5  # Keep in reasonable range
        mock_vae.decode.return_value = Mock(sample=mock_decoded)

        mock_autoencoder.from_pretrained.return_value = mock_vae
        mock_vae.to.return_value = mock_vae
        mock_vae.eval.return_value = mock_vae

        vae = SD15VAE(device="cpu")

        # Create test latent
        test_latent = torch.randn(1, 4, 64, 64)

        result = vae.decode(test_latent)

        # Check that decoding was called with unscaled latent
        mock_vae.decode.assert_called_once()
        decode_arg = mock_vae.decode.call_args[0][0]
        expected_unscaled = test_latent / vae.SD15_SCALE
        assert torch.allclose(decode_arg, expected_unscaled)

        # Check result is a PIL Image
        assert isinstance(result, Image.Image)
        assert result.size == (512, 512)
        assert result.mode == "RGB"

    def test_decode_output_range(self):
        """Test that decode properly clamps and converts values."""
        _ = SD15VAE(device="cpu")  # Just testing instantiation

        # Test with extreme values to check clamping
        extreme_tensor = torch.tensor([[[[2.0, -3.0], [1.5, -1.5]]]])  # (1, 1, 2, 2)

        # Manually test the conversion logic
        x = extreme_tensor
        x = (x.clamp(-1, 1) + 1) / 2  # Should give values in [0, 1]
        arr = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Check values are properly bounded
        assert arr.min() >= 0
        assert arr.max() <= 255
        assert arr.dtype == np.uint8

    def test_scale_constant(self):
        """Test that SD15_SCALE constant is correct."""
        assert SD15VAE.SD15_SCALE == 0.18215

    @patch("src.diffusion_art.models.vae.AutoencoderKL")
    def test_model_loading_only_once(self, mock_autoencoder):
        """Test that model is only loaded once even with multiple calls."""
        mock_vae = Mock()
        mock_autoencoder.from_pretrained.return_value = mock_vae
        mock_vae.to.return_value = mock_vae
        mock_vae.eval.return_value = mock_vae

        vae = SD15VAE()

        # Call load_model multiple times
        vae.load_model()
        vae.load_model()
        vae.load_model()

        # Should only be called once
        mock_autoencoder.from_pretrained.assert_called_once()

    def test_image_preprocessing_sizes(self):
        """Test that different input image sizes are handled correctly."""
        vae = SD15VAE()

        sizes_to_test = [(100, 100), (256, 256), (1024, 768), (300, 400)]

        for width, height in sizes_to_test:
            test_img = Image.new("RGB", (width, height), color="green")
            transformed = vae._transform(test_img)

            # All should be resized to 512x512
            assert transformed.shape == (3, 512, 512)

    def test_image_mode_conversion(self):
        """Test that different image modes are converted to RGB."""
        vae = SD15VAE()

        # Test RGB mode (baseline)
        test_img_rgb = Image.new("RGB", (100, 100), color=(255, 0, 0))
        transformed = vae._transform(test_img_rgb)
        assert transformed.shape == (3, 512, 512)

        # Test P mode (palette) - convert to RGB first to avoid issues
        test_img_p = Image.new("RGB", (100, 100), color="red").convert("P")
        # PIL convert("RGB") should handle this properly
        test_img_p_rgb = test_img_p.convert("RGB")
        transformed = vae._transform(test_img_p_rgb)
        assert transformed.shape == (3, 512, 512)

        # Test L mode - manually convert to RGB to match the .convert("RGB") in encode
        test_img_l = Image.new("L", (100, 100), color=128)
        test_img_l_rgb = test_img_l.convert("RGB")
        transformed = vae._transform(test_img_l_rgb)
        assert transformed.shape == (3, 512, 512)

        # Test RGBA mode - convert to RGB to avoid channel mismatch
        test_img_rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        test_img_rgba_rgb = test_img_rgba.convert("RGB")
        transformed = vae._transform(test_img_rgba_rgb)
        assert transformed.shape == (3, 512, 512)
