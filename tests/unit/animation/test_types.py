"""Tests for animation type definitions and validation."""

import pytest

from diffusion_art.animation.types import (
    AnimationConfig,
    AnimationMetrics,
    AnimationType,
    BreathingPattern,
    RandomWalkType,
)


class TestAnimationConfig:
    """Test animation configuration validation."""

    def test_breathing_config_validation_success(self):
        """Test valid breathing configuration."""
        config = AnimationConfig(
            animation_type=AnimationType.BREATHING,
            breathing_pattern=BreathingPattern.SINE,
            frames=30,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )
        config.validate()  # Should not raise

    def test_breathing_config_missing_pattern(self):
        """Test breathing config fails without breathing_pattern."""
        config = AnimationConfig(
            animation_type=AnimationType.BREATHING,
            breathing_pattern=None,
            frames=30,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )
        with pytest.raises(ValueError, match="breathing_pattern required"):
            config.validate()

    def test_walk_config_validation_success(self):
        """Test valid random walk configuration."""
        config = AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,
            walk_type=RandomWalkType.STANDARD,
            frames=30,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )
        config.validate()  # Should not raise

    def test_walk_config_missing_type(self):
        """Test walk config fails without walk_type."""
        config = AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,
            walk_type=None,
            frames=30,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )
        with pytest.raises(ValueError, match="walk_type required"):
            config.validate()

    def test_invalid_frames(self):
        """Test configuration fails with invalid frame counts."""
        with pytest.raises(ValueError, match="frames must be positive"):
            AnimationConfig(
                animation_type=AnimationType.BREATHING,
                breathing_pattern=BreathingPattern.SINE,
                frames=0,  # Invalid
                fps=24,
                noise_strength=1.0,
                seed=42,
            ).validate()

    def test_invalid_fps(self):
        """Test configuration fails with invalid FPS."""
        with pytest.raises(ValueError, match="fps must be positive"):
            AnimationConfig(
                animation_type=AnimationType.BREATHING,
                breathing_pattern=BreathingPattern.SINE,
                frames=30,
                fps=-1,  # Invalid
                noise_strength=1.0,
                seed=42,
            ).validate()

    def test_invalid_noise_strength(self):
        """Test configuration fails with negative noise strength."""
        with pytest.raises(ValueError, match="noise_strength must be non-negative"):
            AnimationConfig(
                animation_type=AnimationType.BREATHING,
                breathing_pattern=BreathingPattern.SINE,
                frames=30,
                fps=24,
                noise_strength=-0.1,  # Invalid
                seed=42,
            ).validate()

    def test_optional_parameters_defaults(self):
        """Test optional parameters have reasonable defaults."""
        config = AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,
            walk_type=RandomWalkType.MOMENTUM,
            frames=30,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        assert config.momentum == 0.8
        assert config.explore_fraction == 0.62
        assert config.return_home is False


class TestAnimationMetrics:
    """Test animation metrics data structure."""

    def test_metrics_creation(self):
        """Test metrics can be created with all fields."""
        metrics = AnimationMetrics(
            total_frames=100,
            actual_duration_seconds=4.17,
            peak_memory_mb=256.5,
            generation_time_seconds=1.2,
            turn_around_step=50,
        )

        assert metrics.total_frames == 100
        assert metrics.actual_duration_seconds == 4.17
        assert metrics.peak_memory_mb == 256.5
        assert metrics.generation_time_seconds == 1.2
        assert metrics.turn_around_step == 50

    def test_metrics_optional_fields(self):
        """Test metrics work with optional fields."""
        metrics = AnimationMetrics(
            total_frames=60,
            actual_duration_seconds=2.5,
            peak_memory_mb=128.0,
            generation_time_seconds=0.8,
        )

        assert metrics.turn_around_step is None


class TestEnumTypes:
    """Test enum type definitions."""

    def test_breathing_patterns(self):
        """Test all breathing patterns are defined."""
        patterns = list(BreathingPattern)
        pattern_values = [p.value for p in patterns]

        assert "sine" in pattern_values
        assert "heartbeat" in pattern_values
        assert "pulse" in pattern_values
        assert len(patterns) == 3

    def test_random_walk_types(self):
        """Test all random walk types are defined."""
        walk_types = list(RandomWalkType)
        walk_values = [w.value for w in walk_types]

        expected = [
            "distance_threshold",
            "standard",
            "momentum",
            "attracted_home",
            "deep_note",
        ]

        for expected_type in expected:
            assert expected_type in walk_values

        assert len(walk_types) == 5

    def test_animation_types(self):
        """Test animation type enumeration."""
        types = list(AnimationType)
        type_values = [t.value for t in types]

        assert "breathing" in type_values
        assert "random_walk" in type_values
        assert len(types) == 2
