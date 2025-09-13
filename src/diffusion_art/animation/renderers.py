"""Video rendering and preview generation."""

import asyncio
import io
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Iterator, List

import torch
from PIL import Image

from ..models.vae import SD15VAE
from .types import AnimationMetrics


class VideoRenderer:
    """Handles video rendering with async processing."""

    def __init__(self, output_format: str = "mp4", quality_crf: int = 18):
        """Initialize renderer.

        Args:
            output_format: Output format (mp4, webm)
            quality_crf: Video quality (18=high, 23=default, 28=lower)
        """
        self.output_format = output_format
        self.quality_crf = quality_crf

        # Validate ffmpeg availability
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

    async def render_async(self, frames: List[Image.Image], fps: int) -> io.BytesIO:
        """Render video asynchronously to avoid blocking UI."""
        import concurrent.futures

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self._render_sync, frames, fps)

    def _render_sync(self, frames: List[Image.Image], fps: int) -> io.BytesIO:
        """Synchronous video rendering implementation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames as PNG files
            for i, img in enumerate(frames):
                img.save(f"{temp_dir}/frame_{i:06d}.png")

            # Prepare ffmpeg command
            output_path = f"{temp_dir}/animation.{self.output_format}"
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                f"{temp_dir}/frame_%06d.png",
                "-c:v",
                self._get_codec(),
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(self.quality_crf),
                output_path,
            ]

            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            # Read result into buffer
            with open(output_path, "rb") as f:
                return io.BytesIO(f.read())

    def _get_codec(self) -> str:
        """Get appropriate codec for format."""
        codecs = {"mp4": "libx264", "webm": "libvpx-vp9"}
        return codecs.get(self.output_format, "libx264")

    def get_supported_formats(self) -> List[str]:
        """Return list of supported output formats."""
        return ["mp4", "webm"]


class StreamingDecoder:
    """Handles streaming latent -> image decoding to manage memory."""

    def __init__(self, vae_model: SD15VAE, batch_size: int = 4):
        """Initialize decoder.

        Args:
            vae_model: VAE model for decoding
            batch_size: Batch size for decoding (smaller = less memory)
        """
        self.vae_model = vae_model
        self.batch_size = batch_size

    def decode_stream(self, latents: List[torch.Tensor]) -> Iterator[Image.Image]:
        """Stream decode latents in batches to manage memory."""
        for i in range(0, len(latents), self.batch_size):
            batch = latents[i : i + self.batch_size]

            try:
                images = self.vae_model.decode_batch(batch, batch_size=len(batch))
                for img in images:
                    yield img

                # Clean up GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                # Fallback to smaller batch size
                if len(batch) > 1:
                    for latent in batch:
                        yield self.vae_model.decode(latent)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    raise e

    async def decode_stream_async(
        self, latents: List[torch.Tensor]
    ) -> AsyncIterator[Image.Image]:
        """Async version of streaming decoder."""
        for img in self.decode_stream(latents):
            yield img
            # Yield control to event loop
            await asyncio.sleep(0)


class PreviewRenderer:
    """Handles single-frame preview generation."""

    def __init__(self, vae_model: SD15VAE):
        """Initialize preview renderer."""
        self.vae_model = vae_model

    def render_preview(self, latent: torch.Tensor) -> Image.Image:
        """Render single latent frame for preview."""
        return self.vae_model.decode(latent)

    @asynccontextmanager
    async def memory_context(self) -> AsyncGenerator[None, None]:
        """Context manager for memory cleanup during preview."""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
