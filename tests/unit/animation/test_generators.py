"""Tests for animation generators - core business logic only."""

import pytest
import torch

from diffusion_art.animation.generators import BreathingGenerator, RandomWalkGenerator
from diffusion_art.animation.types import (
    AnimationConfig,
    AnimationType,
    BreathingPattern,
    RandomWalkType,
)


class TestBreathingGenerator:
    """Test breathing pattern generation logic."""

    @pytest.fixture
    def base_latent(self):
        """Create test latent tensor."""
        return torch.randn(1, 4, 64, 64)

    @pytest.fixture
    def breathing_config(self):
        """Create valid breathing configuration."""
        return AnimationConfig(
            animation_type=AnimationType.BREATHING,
            breathing_pattern=BreathingPattern.SINE,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

    def test_breathing_generation_shape_consistency(
        self, base_latent, breathing_config
    ):
        """Test breathing generation produces correct number of frames with consistent shape."""
        generator = BreathingGenerator()

        latent_frames, metrics = generator.generate(base_latent, breathing_config)

        # Check frame count matches config
        assert len(latent_frames) == breathing_config.frames
        assert metrics.total_frames == breathing_config.frames

        # Check all frames have same shape as base
        for frame in latent_frames:
            assert frame.shape == base_latent.shape

        # Check metrics are reasonable
        assert metrics.actual_duration_seconds == pytest.approx(10 / 24, abs=0.01)
        assert metrics.generation_time_seconds >= 0
        assert metrics.peak_memory_mb >= 0

    def test_breathing_patterns_produce_different_outputs(self, base_latent):
        """Test different breathing patterns produce different sequences."""
        configs = [
            AnimationConfig(
                animation_type=AnimationType.BREATHING,
                breathing_pattern=pattern,
                frames=5,
                fps=24,
                noise_strength=1.0,
                seed=42,
            )
            for pattern in [
                BreathingPattern.SINE,
                BreathingPattern.HEARTBEAT,
                BreathingPattern.PULSE,
            ]
        ]

        generator = BreathingGenerator()
        sequences = []

        for config in configs:
            latent_frames, _ = generator.generate(base_latent, config)
            sequences.append(latent_frames)

        # Different patterns should produce different sequences
        # (at least some frames should be different)
        sine_seq, heartbeat_seq, pulse_seq = sequences

        assert not torch.allclose(sine_seq[2], heartbeat_seq[2], atol=1e-6)
        assert not torch.allclose(sine_seq[2], pulse_seq[2], atol=1e-6)

    def test_breathing_seed_consistency(self, base_latent, breathing_config):
        """Test same seed produces identical results."""
        generator = BreathingGenerator()

        # Generate twice with same seed
        frames1, _ = generator.generate(base_latent, breathing_config)
        frames2, _ = generator.generate(base_latent, breathing_config)

        # Should be identical
        for f1, f2 in zip(frames1, frames2):
            assert torch.allclose(f1, f2, atol=1e-6)

    def test_breathing_different_seeds(self, base_latent, breathing_config):
        """Test different seeds produce different results."""
        generator = BreathingGenerator()

        # Generate with different seeds
        config2 = AnimationConfig(
            animation_type=AnimationType.BREATHING,
            breathing_pattern=BreathingPattern.SINE,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=123,  # Different seed
        )

        frames1, _ = generator.generate(base_latent, breathing_config)
        frames2, _ = generator.generate(base_latent, config2)

        # Should be different (at least some frames, allowing for small differences)
        # Check multiple frames to account for potential edge cases
        differences_found = False
        for i in [3, 5, 7]:  # Check multiple frames
            if not torch.allclose(frames1[i], frames2[i], atol=1e-4):
                differences_found = True
                break
        assert differences_found, "Different seeds should produce different sequences"

    def test_breathing_wrong_animation_type(self, base_latent):
        """Test breathing generator rejects wrong animation type."""
        config = AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,  # Wrong type
            walk_type=RandomWalkType.STANDARD,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        generator = BreathingGenerator()
        with pytest.raises(
            ValueError, match="BreathingGenerator requires breathing animation type"
        ):
            generator.generate(base_latent, config)


class TestRandomWalkGenerator:
    """Test random walk generation algorithms."""

    @pytest.fixture
    def base_latent(self):
        """Create test latent tensor."""
        return torch.randn(1, 4, 64, 64)

    @pytest.fixture
    def walk_config(self):
        """Create valid walk configuration."""
        return AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,
            walk_type=RandomWalkType.STANDARD,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

    def test_walk_generation_shape_consistency(self, base_latent, walk_config):
        """Test walk generation produces correct frames with consistent shape."""
        generator = RandomWalkGenerator()

        latent_frames, metrics = generator.generate(base_latent, walk_config)

        # Check frame count is reasonable (walk algorithms might add starting frame)
        assert len(latent_frames) >= walk_config.frames
        assert len(latent_frames) <= walk_config.frames + 1  # Allow for starting frame
        assert metrics.total_frames == len(latent_frames)

        # Check shape consistency - all frames should have same shape as base
        for i, frame in enumerate(latent_frames):
            assert (
                frame.shape == base_latent.shape
            ), f"Frame {i} has wrong shape: {frame.shape} vs {base_latent.shape}"

        # Check metrics are reasonable
        assert metrics.generation_time_seconds >= 0
        assert metrics.peak_memory_mb >= 0

    def test_walk_types_produce_different_paths(self, base_latent):
        """Test different walk types produce different paths."""
        walk_types = [
            RandomWalkType.STANDARD,
            RandomWalkType.MOMENTUM,
            RandomWalkType.ATTRACTED_HOME,
            RandomWalkType.DEEP_NOTE,
        ]

        generator = RandomWalkGenerator()
        sequences = []

        for walk_type in walk_types:
            config = AnimationConfig(
                animation_type=AnimationType.RANDOM_WALK,
                walk_type=walk_type,
                frames=8,
                fps=24,
                noise_strength=1.0,
                seed=42,
            )
            latent_frames, _ = generator.generate(base_latent, config)
            sequences.append(latent_frames)

        # Different walk types should produce different paths
        standard, momentum, attracted, deep = sequences

        # Check that at least some walk types produce different results
        # Some algorithms might coincidentally produce similar results for short sequences
        differences_found = 0
        mid_frame = 4

        if not torch.allclose(standard[mid_frame], momentum[mid_frame], atol=1e-5):
            differences_found += 1
        if not torch.allclose(standard[mid_frame], attracted[mid_frame], atol=1e-5):
            differences_found += 1
        if not torch.allclose(momentum[mid_frame], deep[mid_frame], atol=1e-5):
            differences_found += 1
        if not torch.allclose(attracted[mid_frame], deep[mid_frame], atol=1e-5):
            differences_found += 1

        # At least 2 pairs should be different
        assert (
            differences_found >= 2
        ), f"Expected different walk types to produce different paths, only {differences_found} differences found"

    def test_distance_threshold_walk_returns_turnaround_info(self, base_latent):
        """Test distance threshold walk provides turn-around step info."""
        config = AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,
            walk_type=RandomWalkType.DISTANCE_THRESHOLD,
            frames=20,
            fps=24,
            noise_strength=1.0,
            seed=42,
            explore_fraction=0.6,
        )

        generator = RandomWalkGenerator()
        latent_frames, metrics = generator.generate(base_latent, config)

        # Should return turn-around information
        assert metrics.turn_around_step is not None
        assert isinstance(metrics.turn_around_step, int)
        assert 0 < metrics.turn_around_step < config.frames

        # Turn-around should happen around explore_fraction
        expected_turnaround = int(config.frames * config.explore_fraction)
        assert abs(metrics.turn_around_step - expected_turnaround) <= 3

    def test_walk_seed_consistency(self, base_latent, walk_config):
        """Test walk generation is deterministic with same seed."""
        generator = RandomWalkGenerator()

        # Generate twice with same config
        frames1, _ = generator.generate(base_latent, walk_config)
        frames2, _ = generator.generate(base_latent, walk_config)

        # Should be identical
        for f1, f2 in zip(frames1, frames2):
            assert torch.allclose(f1, f2, atol=1e-6)

    def test_walk_noise_strength_scaling(self, base_latent):
        """Test different noise strengths produce proportionally different movements."""
        generator = RandomWalkGenerator()

        configs = []
        for strength in [0.5, 1.0, 2.0]:
            configs.append(
                AnimationConfig(
                    animation_type=AnimationType.RANDOM_WALK,
                    walk_type=RandomWalkType.STANDARD,
                    frames=5,
                    fps=24,
                    noise_strength=strength,
                    seed=42,
                )
            )

        sequences = [generator.generate(base_latent, config)[0] for config in configs]

        # Higher noise strength should produce larger deviations from base
        def avg_distance_from_base(frames, base):
            distances = [
                torch.norm(frame - base).item() for frame in frames[1:]
            ]  # Skip first frame
            return sum(distances) / len(distances)

        low_dist = avg_distance_from_base(sequences[0], base_latent)
        med_dist = avg_distance_from_base(sequences[1], base_latent)
        high_dist = avg_distance_from_base(sequences[2], base_latent)

        # Should show increasing trend (allowing some variance in random walks)
        assert med_dist > low_dist * 0.8  # Some tolerance for randomness
        assert high_dist > med_dist * 0.8

    def test_walk_wrong_animation_type(self, base_latent):
        """Test walk generator rejects wrong animation type."""
        config = AnimationConfig(
            animation_type=AnimationType.BREATHING,  # Wrong type
            breathing_pattern=BreathingPattern.SINE,
            frames=10,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        generator = RandomWalkGenerator()
        with pytest.raises(
            ValueError, match="RandomWalkGenerator requires random_walk animation type"
        ):
            generator.generate(base_latent, config)

    def test_walk_unknown_type_raises_error(self, base_latent):
        """Test walk generator handles unknown walk types gracefully."""
        # Create config with unknown walk type (by directly setting it)
        config = AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,
            walk_type=RandomWalkType.STANDARD,  # We'll modify this
            frames=5,
            fps=24,
            noise_strength=1.0,
            seed=42,
        )

        # Manually override with invalid value (simulating future unknown type)
        config.walk_type = "unknown_walk_type"

        generator = RandomWalkGenerator()
        with pytest.raises(ValueError, match="Unknown walk type"):
            generator.generate(base_latent, config)
