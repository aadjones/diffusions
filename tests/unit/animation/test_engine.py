"""Tests for animation engine orchestration and caching logic."""

from unittest.mock import Mock

import pytest
import torch

from diffusion_art.animation.engine import AnimationEngine
from diffusion_art.animation.types import (
    AnimationConfig,
    AnimationType,
    BreathingPattern,
    RandomWalkType,
)


class TestAnimationEngine:
    """Test animation engine orchestration and caching."""

    @pytest.fixture
    def mock_vae_model(self):
        """Create mock VAE model."""
        mock_vae = Mock()
        mock_vae.decode.return_value = Mock()  # Mock PIL Image
        mock_vae.decode_batch.return_value = [Mock(), Mock()]  # Mock list of PIL Images
        return mock_vae

    @pytest.fixture
    def engine(self, mock_vae_model):
        """Create animation engine with mock VAE."""
        return AnimationEngine(mock_vae_model)

    @pytest.fixture
    def base_latent(self):
        """Create test latent tensor."""
        return torch.randn(1, 4, 64, 64)

    def test_create_breathing_config(self, engine):
        """Test breathing configuration creation helper."""
        config = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,
            frames=30,
            fps=24,
            noise_strength=1.5,
            seed=42,
        )

        assert config.animation_type == AnimationType.BREATHING
        assert config.breathing_pattern == BreathingPattern.SINE
        assert config.frames == 30
        assert config.fps == 24
        assert config.noise_strength == 1.5
        assert config.seed == 42

    def test_create_walk_config(self, engine):
        """Test walk configuration creation helper."""
        config = engine.create_walk_config(
            walk_type=RandomWalkType.MOMENTUM,
            frames=60,
            fps=30,
            noise_strength=2.0,
            seed=123,
            momentum=0.9,
        )

        assert config.animation_type == AnimationType.RANDOM_WALK
        assert config.walk_type == RandomWalkType.MOMENTUM
        assert config.frames == 60
        assert config.fps == 30
        assert config.noise_strength == 2.0
        assert config.seed == 123
        assert config.momentum == 0.9

    def test_preview_sequence_caching(self, engine, base_latent):
        """Test preview sequence caching behavior."""
        config = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,
            frames=50,  # Will be limited to 31 for preview
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        # First generation should create cache entry
        sequence1, metrics1 = engine.generate_preview_sequence(base_latent, config)

        # Second call should use cache (generation_time should be 0)
        sequence2, metrics2 = engine.generate_preview_sequence(base_latent, config)

        # Should return cached result
        assert len(sequence1) == len(sequence2)
        assert metrics2.generation_time_seconds == 0.0  # Cached result
        assert metrics2.peak_memory_mb == 0.0  # Cached result

    def test_preview_sequence_max_frames_limit(self, engine, base_latent):
        """Test preview sequences are limited to reasonable frame counts."""
        config = engine.create_breathing_config(
            pattern=BreathingPattern.PULSE,
            frames=100,  # Large frame count
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        sequence, metrics = engine.generate_preview_sequence(base_latent, config)

        # Should be limited to max_frames (31 by default)
        assert len(sequence) <= 31
        assert metrics.total_frames <= 31

    def test_cache_key_generation_consistency(self, engine, base_latent):
        """Test cache keys are consistent for same inputs."""
        config1 = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,
            frames=20,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        config2 = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,
            frames=20,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        # Generate preview with first config
        engine.generate_preview_sequence(base_latent, config1)

        # Second call with equivalent config should hit cache
        sequence, metrics = engine.generate_preview_sequence(base_latent, config2)
        assert metrics.generation_time_seconds == 0.0  # Cached

    def test_cache_key_differentiation(self, engine, base_latent):
        """Test cache keys differentiate between different configurations."""
        config1 = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,
            frames=20,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        config2 = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,  # Same pattern
            frames=20,
            fps=24,
            noise_strength=1.0,
            seed=123,  # Different seed - guaranteed to be different
        )

        # Generate with first config
        sequence1, metrics1 = engine.generate_preview_sequence(base_latent, config1)

        # Different config should not hit cache and should produce different sequences
        sequence2, metrics2 = engine.generate_preview_sequence(base_latent, config2)

        # Different seeds should produce different sequences
        # Check multiple frames to find differences
        differences_found = False
        for i in [3, 5, 7, 10]:  # Check multiple frames
            if i < len(sequence1) and i < len(sequence2):
                if not torch.allclose(sequence1[i], sequence2[i], atol=1e-5):
                    differences_found = True
                    break

        assert differences_found, "Different seeds should produce different sequences"

    def test_cache_size_limit_enforced(self, engine, base_latent):
        """Test cache size limit is enforced with LRU eviction."""
        # Generate more sequences than cache limit (3)
        configs = []
        for i in range(5):
            config = engine.create_breathing_config(
                pattern=BreathingPattern.SINE,
                frames=20,
                fps=24,
                noise_strength=1.0,
                seed=i,  # Different seeds to avoid cache hits
            )
            configs.append(config)
            engine.generate_preview_sequence(base_latent, config)

        # Cache should be limited to 3 entries
        cache_stats = engine.get_cache_stats()
        assert cache_stats["cached_sequences"] <= 3
        assert cache_stats["cache_size_limit"] == 3

    def test_cache_clearing(self, engine, base_latent):
        """Test cache can be cleared."""
        config = engine.create_breathing_config(
            pattern=BreathingPattern.HEARTBEAT,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        # Generate to populate cache
        engine.generate_preview_sequence(base_latent, config)

        # Verify cache is populated
        stats_before = engine.get_cache_stats()
        assert stats_before["cached_sequences"] > 0

        # Clear cache
        engine.clear_cache()

        # Verify cache is empty
        stats_after = engine.get_cache_stats()
        assert stats_after["cached_sequences"] == 0
        assert stats_after["total_cached_tensors"] == 0

    def test_render_preview_frame_selection(self, engine, base_latent):
        """Test preview rendering selects correct frame from sequence."""
        config = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        # Should not raise and should call VAE decode
        engine.render_preview(base_latent, config, frame_index=5)

        # Should have called vae.decode once
        engine.preview_renderer.vae_model.decode.assert_called_once()

    def test_render_preview_frame_index_bounds_checking(self, engine, base_latent):
        """Test preview frame index is bounded to sequence length."""
        config = engine.create_breathing_config(
            pattern=BreathingPattern.SINE,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        # Frame index beyond sequence length should use last frame
        # Should not raise an error
        engine.render_preview(base_latent, config, frame_index=999)

        # Should still work (use last available frame)
        engine.preview_renderer.vae_model.decode.assert_called_once()

    def test_get_supported_formats(self, engine):
        """Test supported formats are returned."""
        formats = engine.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        # Should at least support mp4
        assert any("mp4" in fmt for fmt in formats)

    def test_cache_stats_structure(self, engine):
        """Test cache stats return expected structure."""
        stats = engine.get_cache_stats()

        required_keys = ["cached_sequences", "total_cached_tensors", "cache_size_limit"]
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], int)
            assert stats[key] >= 0
