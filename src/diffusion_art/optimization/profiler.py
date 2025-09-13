"""Performance profiling for animation rendering pipeline."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ..models.vae import SD15VAE


@dataclass
class PerformanceMetrics:
    """Performance metrics for animation rendering."""

    operation: str
    frames: int
    total_time: float
    time_per_frame: float
    memory_peak_mb: float
    throughput_fps: float

    def __str__(self) -> str:
        return (
            f"{self.operation}: {self.frames} frames in {self.total_time:.2f}s "
            f"({self.time_per_frame:.3f}s/frame, {self.throughput_fps:.1f} FPS, "
            f"{self.memory_peak_mb:.1f}MB peak)"
        )


class AnimationProfiler:
    """Profiles different stages of animation rendering."""

    def __init__(self, vae_model: SD15VAE):
        self.vae_model = vae_model
        self.metrics: List[PerformanceMetrics] = []

    def profile_single_decode(self, num_frames: int = 10) -> PerformanceMetrics:
        """Profile single-frame VAE decoding."""
        latents = [
            torch.randn(1, 4, 64, 64).to(self.vae_model.device)
            for _ in range(num_frames)
        ]

        # Warmup
        self.vae_model.decode(latents[0])

        # Profile
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        start_time = time.time()
        for latent in latents:
            self.vae_model.decode(latent)
        total_time = time.time() - start_time

        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        peak_memory_mb = (peak_memory - initial_memory) / 1024 / 1024

        metrics = PerformanceMetrics(
            operation="Single Decode",
            frames=num_frames,
            total_time=total_time,
            time_per_frame=total_time / num_frames,
            memory_peak_mb=peak_memory_mb,
            throughput_fps=num_frames / total_time,
        )

        self.metrics.append(metrics)
        return metrics

    def profile_batch_decode(
        self, num_frames: int = 16, batch_size: int = 8
    ) -> PerformanceMetrics:
        """Profile batch VAE decoding."""
        latents = [
            torch.randn(1, 4, 64, 64).to(self.vae_model.device)
            for _ in range(num_frames)
        ]

        # Warmup
        self.vae_model.decode_batch(latents[:2], batch_size=2)

        # Profile
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        start_time = time.time()
        _ = self.vae_model.decode_batch(latents, batch_size=batch_size)
        total_time = time.time() - start_time

        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        peak_memory_mb = (peak_memory - initial_memory) / 1024 / 1024

        metrics = PerformanceMetrics(
            operation=f"Batch Decode (bs={batch_size})",
            frames=num_frames,
            total_time=total_time,
            time_per_frame=total_time / num_frames,
            memory_peak_mb=peak_memory_mb,
            throughput_fps=num_frames / total_time,
        )

        self.metrics.append(metrics)
        return metrics

    def find_optimal_batch_size(
        self, test_frames: int = 16
    ) -> Dict[int, PerformanceMetrics]:
        """Find optimal batch size by testing different sizes."""
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}

        for batch_size in batch_sizes:
            try:
                metrics = self.profile_batch_decode(test_frames, batch_size)
                results[batch_size] = metrics
                print(f"✅ {metrics}")
            except Exception as e:
                print(f"❌ Batch size {batch_size} failed: {e}")

        return results

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks and suggest optimizations."""
        if not self.metrics:
            return {"error": "No metrics available"}

        # Find best performing configurations
        best_throughput = max(self.metrics, key=lambda m: m.throughput_fps)
        lowest_memory = min(self.metrics, key=lambda m: m.memory_peak_mb)

        # Generate recommendations first
        recommendations: List[str] = []

        if best_throughput.throughput_fps < 1.0:
            recommendations.append(
                "🐌 Very slow rendering (<1 FPS). Consider model optimization."
            )

        if any(m.memory_peak_mb > 4096 for m in self.metrics):
            recommendations.append(
                "🐏 High memory usage. Consider smaller batch sizes."
            )

        # Calculate potential speedups
        baseline = next((m for m in self.metrics if "Single" in m.operation), None)
        if baseline:
            for metrics in self.metrics:
                if metrics != baseline:
                    speedup = baseline.time_per_frame / metrics.time_per_frame
                    if speedup > 1.1:
                        recommendations.append(
                            f"⚡ {metrics.operation} is {speedup:.1f}x faster than single decode"
                        )

        analysis = {
            "best_throughput": {
                "config": best_throughput.operation,
                "fps": best_throughput.throughput_fps,
                "time_per_frame": best_throughput.time_per_frame,
            },
            "lowest_memory": {
                "config": lowest_memory.operation,
                "memory_mb": lowest_memory.memory_peak_mb,
            },
            "recommendations": recommendations,
        }

        return analysis

    def estimate_video_time(self, frames: int, fps: int = 24) -> Dict[str, Any]:
        """Estimate rendering time for different configurations."""
        if not self.metrics:
            return {"error": "No metrics available"}

        estimates = {}
        for metrics in self.metrics:
            total_seconds = frames * metrics.time_per_frame
            estimates[metrics.operation] = {
                "total_seconds": total_seconds,
                "minutes": total_seconds / 60,
                "video_duration": frames / fps,
            }

        return estimates
