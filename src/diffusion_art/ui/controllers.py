"""UI Controllers for separating presentation from business logic."""

import asyncio
import io
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from PIL import Image

from ..animation import (
    AnimationConfig,
    AnimationEngine,
    AnimationType,
    BreathingPattern,
    RandomWalkType,
)
from ..models.vae import SD15VAE


class BreathingTabController:
    """Controller for breathing tab, separating UI from business logic."""

    def __init__(self, vae_model: SD15VAE):
        """Initialize controller with VAE model."""
        self.vae_model = vae_model
        self.animation_engine = AnimationEngine(vae_model)
        self._current_base_latent: Optional[torch.Tensor] = None

    def encode_base_image(self, image: Image.Image) -> bool:
        """Encode base image to latent space.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._current_base_latent = self.vae_model.encode(image)
            return True
        except Exception:
            self._current_base_latent = None
            return False

    def get_base_latent(self) -> Optional[torch.Tensor]:
        """Get current base latent tensor."""
        return self._current_base_latent

    def create_animation_config(
        self,
        animation_type: str,
        frames: int,
        fps: int,
        noise_strength: float,
        seed: int,
        **kwargs: Any,
    ) -> AnimationConfig:
        """Create animation configuration from UI parameters."""
        # Map UI string values to enum types
        if animation_type in ["sine", "heartbeat", "pulse"]:
            # Breathing pattern
            pattern = BreathingPattern(animation_type)
            return self.animation_engine.create_breathing_config(
                pattern=pattern,
                frames=frames,
                fps=fps,
                noise_strength=noise_strength,
                seed=seed,
            )
        else:
            # Random walk type
            walk_type = RandomWalkType(animation_type)
            return self.animation_engine.create_walk_config(
                walk_type=walk_type,
                frames=frames,
                fps=fps,
                noise_strength=noise_strength,
                seed=seed,
                momentum=kwargs.get("momentum", 0.8),
                explore_fraction=kwargs.get("explore_fraction", 0.62),
                return_home=kwargs.get("return_home", False),
            )

    def generate_preview(
        self, config: AnimationConfig, frame_index: int = 0
    ) -> Tuple[Optional[Image.Image], Optional[str], Optional[Dict[str, Any]]]:
        """Generate preview image.

        Returns:
            Tuple of (preview_image, error_message, metadata)
        """
        if self._current_base_latent is None:
            return None, "No base image encoded", None

        try:
            # Generate preview sequence and get metrics
            sequence, metrics = self.animation_engine.generate_preview_sequence(
                self._current_base_latent, config
            )

            # Get specific frame
            frame_latent = sequence[min(frame_index, len(sequence) - 1)]
            preview_image = self.animation_engine.preview_renderer.render_preview(
                frame_latent
            )

            # Build metadata
            metadata = {
                "total_frames": metrics.total_frames,
                "generation_time": metrics.generation_time_seconds,
                "turn_around_step": metrics.turn_around_step,
                "actual_frame_index": min(frame_index, len(sequence) - 1),
            }

            return preview_image, None, metadata

        except Exception as e:
            return None, f"Preview generation failed: {str(e)}", None

    async def generate_animation_async(
        self,
        config: AnimationConfig,
        keyframe_interval: int = 1,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Optional[bytes], Optional[str], Optional[Dict[str, Any]]]:
        """Generate full animation asynchronously.

        Args:
            config: Animation configuration
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (video_bytes, error_message, metrics)
        """
        if self._current_base_latent is None:
            return None, "No base image encoded", None

        try:
            if progress_callback:
                progress_callback(0.1, "Generating latent sequence...")

            # Generate full animation
            images, metrics = await self.animation_engine.generate_full_animation(
                self._current_base_latent, config, keyframe_interval=keyframe_interval
            )

            if progress_callback:
                progress_callback(0.7, "Rendering video...")

            # Render to video
            video_bytes = await self.animation_engine.render_video_async(
                images, config.fps, "mp4"
            )

            if progress_callback:
                progress_callback(1.0, "Animation complete!")

            # Convert metrics to dict for JSON serialization
            metrics_dict = {
                "total_frames": metrics.total_frames,
                "duration_seconds": metrics.actual_duration_seconds,
                "generation_time_seconds": metrics.generation_time_seconds,
                "peak_memory_mb": metrics.peak_memory_mb,
                "turn_around_step": metrics.turn_around_step,
            }

            return video_bytes, None, metrics_dict

        except Exception as e:
            return None, f"Animation generation failed: {str(e)}", None

    def get_animation_types(self) -> Dict[str, list]:
        """Get available animation types for UI."""
        return {
            "breathing": ["sine", "heartbeat", "pulse"],
            "random_walk": [
                "distance_threshold",
                "standard",
                "momentum",
                "attracted_home",
                "deep_note",
            ],
        }

    def get_supported_formats(self) -> list:
        """Get supported video formats."""
        return self.animation_engine.get_supported_formats()

    def clear_cache(self) -> None:
        """Clear animation cache to free memory."""
        self.animation_engine.clear_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.animation_engine.get_cache_stats()
