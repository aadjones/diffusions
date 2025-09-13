"""Animation generators for latent space sequences."""

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from ..core.noise import breathing_animation
from ..core.random_walk import (
    deep_note_walk,
    distance_threshold_walk,
    latent_random_walk,
    momentum_random_walk,
)
from .types import (
    AnimationConfig,
    AnimationMetrics,
    AnimationType,
    BreathingPattern,
    RandomWalkType,
)


class BaseGenerator(ABC):
    """Base class for latent sequence generators."""

    @abstractmethod
    def generate(
        self, base_latent: torch.Tensor, config: AnimationConfig
    ) -> Tuple[List[torch.Tensor], AnimationMetrics]:
        """Generate latent sequence and return metrics."""
        pass


class BreathingGenerator(BaseGenerator):
    """Generator for breathing pattern animations."""

    def generate(
        self, base_latent: torch.Tensor, config: AnimationConfig
    ) -> Tuple[List[torch.Tensor], AnimationMetrics]:
        """Generate breathing animation sequence."""
        if config.animation_type != AnimationType.BREATHING:
            raise ValueError("BreathingGenerator requires breathing animation type")

        config.validate()
        start_time = time.time()
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Generate breathing animation
        if config.breathing_pattern is None:
            raise ValueError("breathing_pattern is required for breathing animations")

        latent_frames = breathing_animation(
            base_latent,
            frames=config.frames,
            max_strength=config.noise_strength,
            pattern=config.breathing_pattern.value,
            seed=config.seed,
        )

        end_time = time.time()
        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        peak_memory_mb = (peak_memory - initial_memory) / 1024 / 1024

        metrics = AnimationMetrics(
            total_frames=len(latent_frames),
            actual_duration_seconds=len(latent_frames) / config.fps,
            peak_memory_mb=peak_memory_mb,
            generation_time_seconds=end_time - start_time,
        )

        return latent_frames, metrics


class RandomWalkGenerator(BaseGenerator):
    """Generator for random walk animations."""

    def generate(
        self, base_latent: torch.Tensor, config: AnimationConfig
    ) -> Tuple[List[torch.Tensor], AnimationMetrics]:
        """Generate random walk animation sequence."""
        if config.animation_type != AnimationType.RANDOM_WALK:
            raise ValueError("RandomWalkGenerator requires random_walk animation type")

        config.validate()
        start_time = time.time()
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Generate based on walk type
        turn_around_step: Optional[int] = None

        if config.walk_type == RandomWalkType.DISTANCE_THRESHOLD:
            latent_frames, turn_around_step = distance_threshold_walk(
                base_latent,
                steps=config.frames,
                step_size=config.noise_strength,
                explore_fraction=config.explore_fraction,
                seed=config.seed,
            )
        elif config.walk_type == RandomWalkType.STANDARD:
            latent_frames = latent_random_walk(
                base_latent,
                steps=config.frames,
                step_size=config.noise_strength,
                seed=config.seed,
            )
        elif config.walk_type == RandomWalkType.MOMENTUM:
            latent_frames = momentum_random_walk(
                base_latent,
                steps=config.frames,
                step_size=config.noise_strength,
                momentum=config.momentum,
                seed=config.seed,
            )
        elif config.walk_type == RandomWalkType.ATTRACTED_HOME:
            latent_frames = latent_random_walk(
                base_latent,
                steps=config.frames,
                step_size=config.noise_strength,
                return_home=config.return_home,
                seed=config.seed,
            )
        elif config.walk_type == RandomWalkType.DEEP_NOTE:
            latent_frames = deep_note_walk(
                base_latent,
                steps=config.frames,
                step_size=config.noise_strength,
                seed=config.seed,
            )
        else:
            raise ValueError(f"Unknown walk type: {config.walk_type}")

        end_time = time.time()
        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        peak_memory_mb = (peak_memory - initial_memory) / 1024 / 1024

        metrics = AnimationMetrics(
            total_frames=len(latent_frames),
            actual_duration_seconds=len(latent_frames) / config.fps,
            peak_memory_mb=peak_memory_mb,
            generation_time_seconds=end_time - start_time,
            turn_around_step=turn_around_step,
        )

        return latent_frames, metrics
