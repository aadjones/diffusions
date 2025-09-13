"""Type definitions for animation system."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BreathingPattern(Enum):
    """Breathing animation patterns."""

    SINE = "sine"
    HEARTBEAT = "heartbeat"
    PULSE = "pulse"


class RandomWalkType(Enum):
    """Random walk animation types."""

    DISTANCE_THRESHOLD = "distance_threshold"
    STANDARD = "standard"
    MOMENTUM = "momentum"
    ATTRACTED_HOME = "attracted_home"
    DEEP_NOTE = "deep_note"


class AnimationType(Enum):
    """Top-level animation types."""

    BREATHING = "breathing"
    RANDOM_WALK = "random_walk"


@dataclass
class AnimationConfig:
    """Configuration for animation generation."""

    # Core parameters
    animation_type: AnimationType
    frames: int
    fps: int
    noise_strength: float
    seed: int

    # Type-specific parameters
    breathing_pattern: Optional[BreathingPattern] = None
    walk_type: Optional[RandomWalkType] = None

    # Optional parameters
    momentum: float = 0.8  # For momentum walks
    explore_fraction: float = 0.62  # For distance threshold walks
    return_home: bool = False  # For attracted home walks

    def validate(self) -> None:
        """Validate configuration consistency."""
        if (
            self.animation_type == AnimationType.BREATHING
            and self.breathing_pattern is None
        ):
            raise ValueError("breathing_pattern required for breathing animations")

        if self.animation_type == AnimationType.RANDOM_WALK and self.walk_type is None:
            raise ValueError("walk_type required for random walk animations")

        if self.frames <= 0:
            raise ValueError("frames must be positive")

        if self.fps <= 0:
            raise ValueError("fps must be positive")

        if self.noise_strength < 0:
            raise ValueError("noise_strength must be non-negative")


@dataclass
class AnimationMetrics:
    """Metrics from animation generation."""

    total_frames: int
    actual_duration_seconds: float
    peak_memory_mb: float
    generation_time_seconds: float
    turn_around_step: Optional[int] = None  # For distance threshold walks
