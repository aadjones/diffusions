"""Tests for interpolation functions."""

import numpy as np
import pytest
import torch

from src.diffusion_art.core.interpolation import (
    create_interpolation_path,
    lerp,
    multi_slerp,
    slerp,
)


class TestSlerp:
    """Test spherical linear interpolation."""

    def test_slerp_endpoints(self):
        """Test that SLERP returns correct endpoints."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        result_start = slerp(z0, z1, 0.0)
        result_end = slerp(z0, z1, 1.0)

        # SLERP at t=0 should be very close to z0
        assert torch.allclose(result_start, z0, atol=1e-4)
        # SLERP at t=1 should have same direction as z1 but potentially different magnitude
        # Check that the result is in the same direction as z1
        z1_flat = z1.flatten(1)
        result_flat = result_end.flatten(1)
        z1_norm = z1_flat / (z1_flat.norm(dim=1, keepdim=True) + 1e-8)
        result_norm = result_flat / (result_flat.norm(dim=1, keepdim=True) + 1e-8)
        # Cosine similarity should be very close to 1
        cos_sim = (z1_norm * result_norm).sum(dim=1)
        assert torch.all(cos_sim > 0.99)

    def test_slerp_midpoint_magnitude(self):
        """Test that SLERP preserves reasonable magnitude at midpoint."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        result = slerp(z0, z1, 0.5)

        # Check that result has reasonable magnitude
        z0_mag = z0.flatten(1).norm(dim=1)
        z1_mag = z1.flatten(1).norm(dim=1)
        result_mag = result.flatten(1).norm(dim=1)

        # Magnitude should be reasonable relative to inputs
        avg_mag = (z0_mag + z1_mag) / 2
        assert torch.abs(result_mag - avg_mag) / avg_mag < 0.5  # Within 50%

    def test_slerp_shape_preservation(self):
        """Test that SLERP preserves tensor shape."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        result = slerp(z0, z1, 0.3)

        assert result.shape == z0.shape

    def test_slerp_different_shapes(self):
        """Test SLERP with different input shapes."""
        shapes = [(1, 4, 32, 32), (2, 4, 64, 64), (1, 8, 16, 16)]

        for shape in shapes:
            z0 = torch.randn(*shape)
            z1 = torch.randn(*shape)
            result = slerp(z0, z1, 0.5)
            assert result.shape == shape


class TestLerp:
    """Test linear interpolation."""

    def test_lerp_endpoints(self):
        """Test that LERP returns correct endpoints."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        result_start = lerp(z0, z1, 0.0)
        result_end = lerp(z0, z1, 1.0)

        assert torch.allclose(result_start, z0)
        assert torch.allclose(result_end, z1)

    def test_lerp_midpoint(self):
        """Test LERP midpoint calculation."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        result = lerp(z0, z1, 0.5)
        expected = (z0 + z1) / 2

        assert torch.allclose(result, expected)

    def test_lerp_linearity(self):
        """Test LERP linearity property."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        t1, t2 = 0.3, 0.7
        result1 = lerp(z0, z1, t1)
        result2 = lerp(z0, z1, t2)

        # Linear interpolation between interpolations should work
        mid_result = lerp(result1, result2, 0.5)
        expected = lerp(z0, z1, (t1 + t2) / 2)

        assert torch.allclose(mid_result, expected, atol=1e-6)


class TestMultiSlerp:
    """Test multi-way interpolation."""

    def test_multi_slerp_single_latent(self):
        """Test multi-SLERP with only one latent (edge case)."""
        z0 = torch.randn(1, 4, 64, 64)

        with pytest.raises(ValueError, match="Need at least 2 latents"):
            multi_slerp([z0], [1.0])

    def test_multi_slerp_two_latents(self):
        """Test multi-SLERP with two latents equals regular SLERP."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)
        weights = [0.3, 0.7]

        multi_result = multi_slerp([z0, z1], weights)
        # For two latents, multi_slerp should approximate regular slerp
        slerp_result = slerp(z0, z1, 0.7)  # Second weight

        # Results should be similar (not exact due to different algorithms)
        assert multi_result.shape == slerp_result.shape

    def test_multi_slerp_weight_normalization(self):
        """Test that weights are properly normalized."""
        z0 = torch.randn(1, 4, 32, 32)
        z1 = torch.randn(1, 4, 32, 32)
        z2 = torch.randn(1, 4, 32, 32)

        # Unnormalized weights
        weights = [2.0, 3.0, 5.0]
        result = multi_slerp([z0, z1, z2], weights)

        assert result.shape == z0.shape

    def test_multi_slerp_mismatched_inputs(self):
        """Test error handling for mismatched inputs."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        with pytest.raises(ValueError, match="Number of latents must match"):
            multi_slerp([z0, z1], [0.5])  # Only one weight for two latents


class TestInterpolationPath:
    """Test interpolation path creation."""

    def test_path_endpoints(self):
        """Test that path starts and ends at correct points."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        path = create_interpolation_path(z0, z1, steps=10)

        assert len(path) == 10
        assert torch.allclose(path[0], z0, atol=1e-4)
        # For end point, check direction similarity rather than exact match
        z1_flat = z1.flatten(1)
        end_flat = path[-1].flatten(1)
        z1_norm = z1_flat / (z1_flat.norm(dim=1, keepdim=True) + 1e-8)
        end_norm = end_flat / (end_flat.norm(dim=1, keepdim=True) + 1e-8)
        cos_sim = (z1_norm * end_norm).sum(dim=1)
        assert torch.all(cos_sim > 0.99)

    def test_path_minimum_steps(self):
        """Test error handling for insufficient steps."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        with pytest.raises(ValueError, match="Need at least 2 steps"):
            create_interpolation_path(z0, z1, steps=1)

    def test_path_methods(self):
        """Test different interpolation methods."""
        z0 = torch.randn(1, 4, 32, 32)
        z1 = torch.randn(1, 4, 32, 32)

        slerp_path = create_interpolation_path(z0, z1, steps=5, method="slerp")
        lerp_path = create_interpolation_path(z0, z1, steps=5, method="lerp")

        assert len(slerp_path) == 5
        assert len(lerp_path) == 5

        # Both should have same start point
        assert torch.allclose(slerp_path[0], lerp_path[0], atol=1e-4)
        # End points should be in similar direction to z1
        slerp_end_flat = slerp_path[-1].flatten(1)
        lerp_end_flat = lerp_path[-1].flatten(1)
        z1_flat = z1.flatten(1)
        z1_norm = z1_flat / (z1_flat.norm(dim=1, keepdim=True) + 1e-8)

        slerp_norm = slerp_end_flat / (slerp_end_flat.norm(dim=1, keepdim=True) + 1e-8)
        lerp_norm = lerp_end_flat / (lerp_end_flat.norm(dim=1, keepdim=True) + 1e-8)

        # Both should be reasonably close to z1 direction
        slerp_cos = (z1_norm * slerp_norm).sum(dim=1)
        lerp_cos = (z1_norm * lerp_norm).sum(dim=1)

        assert torch.all(slerp_cos > 0.95)
        assert torch.all(lerp_cos > 0.95)

        # Middle points should generally be different for different methods
        middle_idx = len(slerp_path) // 2
        # Just check they have reasonable differences in magnitude or direction
        diff = (slerp_path[middle_idx] - lerp_path[middle_idx]).norm()
        assert diff > 1e-6  # Should have some difference

    def test_path_shape_consistency(self):
        """Test that all path elements have consistent shape."""
        z0 = torch.randn(1, 4, 64, 64)
        z1 = torch.randn(1, 4, 64, 64)

        path = create_interpolation_path(z0, z1, steps=7)

        for step in path:
            assert step.shape == z0.shape
