"""VAE optimization strategies for faster decoding."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..models.vae import SD15VAE


class VAEOptimizer:
    """Optimizes VAE decoding for better performance."""

    def __init__(self, vae_model: SD15VAE):
        self.vae_model = vae_model
        self.original_decode = vae_model.decode
        self.original_decode_batch = vae_model.decode_batch

    def enable_fp16(self) -> None:
        """Enable FP16 precision for 2x speedup (if supported)."""
        self.vae_model.load_model()
        if self.vae_model.vae is not None:
            # Convert model to half precision
            self.vae_model.vae = self.vae_model.vae.half()
            print("✅ Enabled FP16 precision")

    def optimize_memory(self) -> None:
        """Enable memory optimizations."""
        if torch.cuda.is_available():
            # Enable memory efficient attention if available
            try:
                torch.backends.cudnn.benchmark = True
                print("✅ Enabled cuDNN benchmark")
            except Exception:
                pass

    def create_fast_decode_batch(self) -> None:
        """Create optimized batch decode that avoids tensor stacking overhead."""

        def fast_decode_batch(
            latent_batch: List[torch.Tensor], batch_size: int = 8
        ) -> List[Image.Image]:
            self.vae_model.load_model()
            assert self.vae_model.vae is not None

            images = []
            with torch.no_grad():
                for i in range(0, len(latent_batch), batch_size):
                    batch = latent_batch[i : i + batch_size]

                    # Efficient stacking - avoid intermediate lists
                    if len(batch) == 1:
                        batch_tensor = batch[0]
                    else:
                        batch_tensor = torch.cat(batch, dim=0)

                    # Decode with proper error handling
                    try:
                        decoded = self.vae_model.vae.decode(
                            batch_tensor / self.vae_model.SD15_SCALE
                        )  # pyright: ignore
                        x = decoded.sample  # pyright: ignore
                        x = (x.clamp(-1, 1) + 1) / 2

                        # Batch convert to images
                        for j in range(x.shape[0]):
                            arr = (x[j].permute(1, 2, 0).cpu().numpy() * 255).astype(
                                np.uint8
                            )
                            images.append(Image.fromarray(arr))

                    except Exception as e:
                        print(
                            f"Batch decode failed: {e}, falling back to individual decodes"
                        )
                        # Fallback to individual decodes
                        for latent in batch:
                            img = self.original_decode(latent)
                            images.append(img)

                    # Clear cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return images

        # Replace the original method
        self.vae_model.decode_batch = fast_decode_batch  # type: ignore

    def create_resolution_scaled_decode(self, target_size: int = 256) -> None:
        """Create a decode method that renders at lower resolution then upscales."""
        _ = self.original_decode

        def scaled_decode(latent: torch.Tensor) -> Image.Image:
            """Decode at lower resolution and upscale for speed."""
            self.vae_model.load_model()
            assert self.vae_model.vae is not None

            with torch.no_grad():
                # Decode normally
                decoded = self.vae_model.vae.decode(
                    latent / self.vae_model.SD15_SCALE
                )  # pyright: ignore
                x = decoded.sample  # pyright: ignore
                x = (x.clamp(-1, 1) + 1) / 2

                # Convert to PIL at original size first
                arr = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(arr)

                # If target size is smaller, resize down then back up
                if target_size < 512:
                    img = img.resize((target_size, target_size), Image.LANCZOS)
                    img = img.resize((512, 512), Image.LANCZOS)

                return img

        # This doesn't actually speed up the VAE decode, but demonstrates the pattern
        # A real implementation would modify the VAE architecture
        print(f"⚠️ Resolution scaling demo created (target: {target_size}px)")

    def benchmark_optimizations(self, test_frames: int = 8) -> dict:
        """Benchmark different optimization strategies."""
        from .profiler import AnimationProfiler

        results = {}
        profiler = AnimationProfiler(self.vae_model)

        # Baseline performance
        print("🔄 Testing baseline performance...")
        baseline = profiler.profile_batch_decode(test_frames, batch_size=4)
        results["baseline"] = baseline

        # Test fast batch decode
        print("🔄 Testing optimized batch decode...")
        self.create_fast_decode_batch()
        optimized = profiler.profile_batch_decode(test_frames, batch_size=4)
        results["optimized_batch"] = optimized

        # Calculate improvements
        speedup = baseline.time_per_frame / optimized.time_per_frame
        print(f"⚡ Optimized batch decode: {speedup:.2f}x speedup")

        return results

    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for further optimization."""
        recommendations = [
            "🔥 **Model Compilation**: Use torch.compile() for 20-30% speedup (PyTorch 2.0+)",
            "🚀 **TensorRT/CoreML**: Convert model for 3-5x speedup on GPU/Apple Silicon",
            "📉 **Lower Precision**: FP16 can provide 2x speedup with minimal quality loss",
            "🎯 **Batch Optimization**: Find optimal batch size for your hardware",
            "💾 **Memory Management**: Use gradient checkpointing to trade compute for memory",
            "🔄 **Pipeline Parallelism**: Overlap VAE decode with video encoding",
            "📱 **Mobile Optimization**: Use quantized models for edge devices",
        ]

        # Hardware-specific recommendations
        device = self.vae_model.device
        if device == "mps":
            recommendations.append(
                "🍎 **Apple Silicon**: Consider using CoreML converted models"
            )
        elif "cuda" in device:
            recommendations.append(
                "🔥 **CUDA**: Enable Tensor Cores with mixed precision"
            )

        return recommendations

    def estimate_theoretical_limits(self) -> dict:
        """Estimate theoretical performance limits."""
        # VAE decoder theoretical analysis
        latent_elements = 1 * 4 * 64 * 64  # 16,384 elements
        output_elements = 1 * 3 * 512 * 512  # 786,432 elements
        upscale_factor = output_elements / latent_elements  # ~48x

        return {
            "latent_tensor_size": f"{latent_elements:,} elements",
            "output_tensor_size": f"{output_elements:,} elements",
            "complexity_ratio": f"{upscale_factor:.1f}x",
            "theoretical_limits": {
                "memory_bound": "~2GB VRAM for batch_size=16",
                "compute_bound": "Depends on model parameters (~100M+)",
                "io_bound": "PNG save/ffmpeg typically <10% of total time",
            },
            "optimization_ceiling": {
                "fp16": "~2x speedup",
                "tensorrt": "~3-5x speedup",
                "optimal_batching": "~1.5-2x speedup",
                "model_pruning": "~2-3x speedup (with quality loss)",
            },
        }

    def reset_optimizations(self) -> None:
        """Reset all optimizations and restore original methods."""
        self.vae_model.decode = self.original_decode  # type: ignore
        self.vae_model.decode_batch = self.original_decode_batch  # type: ignore
        print("🔄 Reset all optimizations")
