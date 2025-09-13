"""Main animation engine orchestrating generators and renderers."""

import asyncio
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from ..models.vae import SD15VAE
from .generators import BreathingGenerator, RandomWalkGenerator
from .renderers import PreviewRenderer, StreamingDecoder, VideoRenderer
from .types import (
    AnimationConfig,
    AnimationMetrics,
    AnimationType,
    BreathingPattern,
    RandomWalkType,
)


class AnimationEngine:
    """Main engine for latent space animation generation."""

    def __init__(self, vae_model: SD15VAE):
        """Initialize animation engine.

        Args:
            vae_model: VAE model for encoding/decoding
        """
        self.vae_model = vae_model
        self.breathing_generator = BreathingGenerator()
        self.walk_generator = RandomWalkGenerator()
        self.video_renderer = VideoRenderer()
        self.streaming_decoder = StreamingDecoder(vae_model)
        self.preview_renderer = PreviewRenderer(vae_model)

        # Cache for preview sequences (LRU with size limit)
        self._preview_cache_size = 3
        self._preview_cache: Dict[str, List[torch.Tensor]] = {}

    def create_breathing_config(
        self,
        pattern: BreathingPattern,
        frames: int,
        fps: int,
        noise_strength: float,
        seed: int,
    ) -> AnimationConfig:
        """Create configuration for breathing animation."""
        return AnimationConfig(
            animation_type=AnimationType.BREATHING,
            breathing_pattern=pattern,
            frames=frames,
            fps=fps,
            noise_strength=noise_strength,
            seed=seed,
        )

    def create_walk_config(
        self,
        walk_type: RandomWalkType,
        frames: int,
        fps: int,
        noise_strength: float,
        seed: int,
        momentum: float = 0.8,
        explore_fraction: float = 0.62,
        return_home: bool = False,
    ) -> AnimationConfig:
        """Create configuration for random walk animation."""
        return AnimationConfig(
            animation_type=AnimationType.RANDOM_WALK,
            walk_type=walk_type,
            frames=frames,
            fps=fps,
            noise_strength=noise_strength,
            seed=seed,
            momentum=momentum,
            explore_fraction=explore_fraction,
            return_home=return_home,
        )

    def generate_preview_sequence(
        self, base_latent: torch.Tensor, config: AnimationConfig, max_frames: int = 31
    ) -> Tuple[List[torch.Tensor], AnimationMetrics]:
        """Generate short sequence for preview purposes."""
        # Create preview config with limited frames
        preview_config = AnimationConfig(
            animation_type=config.animation_type,
            frames=min(config.frames, max_frames),
            fps=config.fps,
            noise_strength=config.noise_strength,
            seed=config.seed,
            breathing_pattern=config.breathing_pattern,
            walk_type=config.walk_type,
            momentum=config.momentum,
            explore_fraction=config.explore_fraction,
            return_home=config.return_home,
        )

        # Use cache for previews
        cache_key = self._get_cache_key(base_latent, preview_config)
        if cache_key in self._preview_cache:
            # Return cached sequence with simple metrics
            cached_sequence = self._preview_cache[cache_key]
            metrics = AnimationMetrics(
                total_frames=len(cached_sequence),
                actual_duration_seconds=len(cached_sequence) / config.fps,
                peak_memory_mb=0.0,  # Cached, no generation
                generation_time_seconds=0.0,
            )
            return cached_sequence, metrics

        # Generate new sequence
        if config.animation_type == AnimationType.BREATHING:
            latents, metrics = self.breathing_generator.generate(
                base_latent, preview_config
            )
        else:
            latents, metrics = self.walk_generator.generate(base_latent, preview_config)

        # Cache the result
        self._cache_sequence(cache_key, latents)

        return latents, metrics

    def render_preview(
        self, base_latent: torch.Tensor, config: AnimationConfig, frame_index: int = 0
    ) -> Image.Image:
        """Render single frame for preview."""
        sequence, _ = self.generate_preview_sequence(base_latent, config)
        frame_latent = sequence[min(frame_index, len(sequence) - 1)]
        return self.preview_renderer.render_preview(frame_latent)

    async def generate_full_animation(
        self,
        base_latent: torch.Tensor,
        config: AnimationConfig,
        keyframe_interval: int = 1,
    ) -> Tuple[List[Image.Image], AnimationMetrics]:
        """Generate complete animation sequence."""
        config.validate()

        # Generate latent sequence
        if config.animation_type == AnimationType.BREATHING:
            latents, metrics = self.breathing_generator.generate(base_latent, config)
        else:
            latents, metrics = self.walk_generator.generate(base_latent, config)

        # Convert to images - use keyframe rendering if interval > 1
        if keyframe_interval > 1:
            from ..optimization import KeyframeRenderer

            keyframe_renderer = KeyframeRenderer(self.vae_model)
            images = keyframe_renderer.render_with_keyframes(
                latents, keyframe_interval=keyframe_interval
            )
        else:
            # Use streaming decoder for full quality
            images = []
            async for img in self.streaming_decoder.decode_stream_async(latents):
                images.append(img)

        return images, metrics

    async def render_video_async(
        self, images: List[Image.Image], fps: int, output_format: str = "mp4"
    ) -> bytes:
        """Render images to video format asynchronously."""
        self.video_renderer.output_format = output_format
        buffer = await self.video_renderer.render_async(images, fps)
        return buffer.getvalue()

    def get_supported_formats(self) -> List[str]:
        """Get supported output formats."""
        return self.video_renderer.get_supported_formats()

    def _get_cache_key(self, base_latent: torch.Tensor, config: AnimationConfig) -> str:
        """Generate cache key for latent sequence."""
        # Use hash of latent tensor content instead of memory id
        latent_hash = hash(base_latent.cpu().numpy().tobytes())
        return f"{latent_hash}_{config.animation_type.value}_{config.noise_strength}_{config.seed}"

    def _cache_sequence(self, key: str, sequence: List[torch.Tensor]) -> None:
        """Cache sequence with LRU eviction."""
        # Remove oldest entries if cache is full
        while len(self._preview_cache) >= self._preview_cache_size:
            oldest_key = next(iter(self._preview_cache))
            del self._preview_cache[oldest_key]

        self._preview_cache[key] = sequence

    def clear_cache(self) -> None:
        """Clear preview cache to free memory."""
        self._preview_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging."""
        total_tensors = sum(len(seq) for seq in self._preview_cache.values())
        return {
            "cached_sequences": len(self._preview_cache),
            "total_cached_tensors": total_tensors,
            "cache_size_limit": self._preview_cache_size,
        }
