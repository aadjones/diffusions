"""Keyframe-based rendering for significant speedups."""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ..models.vae import SD15VAE


class KeyframeRenderer:
    """Renders animations using keyframe interpolation for speed."""

    def __init__(self, vae_model: SD15VAE):
        self.vae_model = vae_model

    def render_with_keyframes(
        self,
        latent_sequence: List[torch.Tensor],
        keyframe_interval: int = 8,
        interpolation_method: str = "linear",
    ) -> List[Image.Image]:
        """
        Render animation using keyframe interpolation.

        Args:
            latent_sequence: Full sequence of latent tensors
            keyframe_interval: Decode every Nth frame
            interpolation_method: 'linear' or 'cubic'

        Returns:
            List of interpolated images
        """
        total_frames = len(latent_sequence)

        # Determine keyframe indices
        keyframe_indices = list(range(0, total_frames, keyframe_interval))
        if keyframe_indices[-1] != total_frames - 1:
            keyframe_indices.append(total_frames - 1)

        print(
            f"🔑 Rendering {len(keyframe_indices)} keyframes out of {total_frames} total frames"
        )

        # Decode keyframes only
        keyframes = {}
        for idx in keyframe_indices:
            print(f"  Decoding keyframe {idx}")
            keyframes[idx] = self.vae_model.decode(latent_sequence[idx])

        # Interpolate between keyframes
        interpolated_images = []

        for frame_idx in range(total_frames):
            if frame_idx in keyframes:
                # Use actual keyframe
                interpolated_images.append(keyframes[frame_idx])
            else:
                # Interpolate between surrounding keyframes
                prev_key = max(k for k in keyframe_indices if k < frame_idx)
                next_key = min(k for k in keyframe_indices if k > frame_idx)

                # Calculate interpolation weight
                alpha = (frame_idx - prev_key) / (next_key - prev_key)

                # Interpolate in image space
                interp_img = self._interpolate_images(
                    keyframes[prev_key],
                    keyframes[next_key],
                    alpha,
                    method=interpolation_method,
                )
                interpolated_images.append(interp_img)

        return interpolated_images

    def _interpolate_images(
        self, img1: Image.Image, img2: Image.Image, alpha: float, method: str = "linear"
    ) -> Image.Image:
        """Interpolate between two images."""
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        if method == "linear":
            # Linear interpolation
            interp_arr = (1 - alpha) * arr1 + alpha * arr2
        elif method == "cubic":
            # Cubic interpolation (smoother)
            # Using a simple cubic ease-in-out curve
            cubic_alpha = self._cubic_ease_in_out(alpha)
            interp_arr = (1 - cubic_alpha) * arr1 + cubic_alpha * arr2
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Clamp values and convert back
        interp_arr = np.clip(interp_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(interp_arr)

    def _cubic_ease_in_out(self, t: float) -> float:
        """Cubic ease-in-out interpolation curve."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def analyze_quality_vs_speed_tradeoff(
        self,
        latent_sequence: List[torch.Tensor],
        test_intervals: List[int] = [2, 4, 8, 16],
    ) -> dict:
        """Analyze quality vs speed tradeoff for different keyframe intervals."""
        results = {}

        # Render ground truth (decode everything)
        print("📊 Rendering ground truth...")
        import time

        start_time = time.time()
        ground_truth = [
            self.vae_model.decode(latent) for latent in latent_sequence[:16]
        ]  # Limit for testing
        ground_truth_time = time.time() - start_time

        for interval in test_intervals:
            if interval >= len(latent_sequence):
                continue

            print(f"Testing keyframe interval: {interval}")
            start_time = time.time()

            # Render with keyframes
            keyframe_result = self.render_with_keyframes(
                latent_sequence[:16], keyframe_interval=interval  # Limit for testing
            )
            keyframe_time = time.time() - start_time

            # Calculate quality metric (MSE with ground truth)
            total_mse = 0.0
            for i, (gt_img, kf_img) in enumerate(zip(ground_truth, keyframe_result)):
                gt_arr = np.array(gt_img, dtype=np.float32)
                kf_arr = np.array(kf_img, dtype=np.float32)
                mse = np.mean((gt_arr - kf_arr) ** 2)
                total_mse += float(mse)

            avg_mse = total_mse / len(ground_truth)
            speedup = ground_truth_time / keyframe_time

            results[interval] = {
                "speedup": speedup,
                "quality_mse": avg_mse,
                "render_time": keyframe_time,
                "frames_decoded": len(range(0, 16, interval))
                + (1 if 15 % interval != 0 else 0),
            }

            print(
                f"  Speedup: {speedup:.2f}x, MSE: {avg_mse:.1f}, Decoded: {results[interval]['frames_decoded']}/16 frames"
            )

        return results

    def estimate_production_speedup(
        self, total_frames: int, keyframe_interval: int = 8
    ) -> dict:
        """Estimate speedup for production-size animations."""
        frames_to_decode = len(range(0, total_frames, keyframe_interval))
        if (total_frames - 1) % keyframe_interval != 0:
            frames_to_decode += 1

        decode_ratio = frames_to_decode / total_frames
        theoretical_speedup = 1 / decode_ratio

        # Account for interpolation overhead (usually negligible)
        interpolation_overhead = 0.02  # 20ms per frame
        decode_time_per_frame = 2.0  # 2s per frame from our benchmarks

        original_time = total_frames * decode_time_per_frame
        keyframe_time = frames_to_decode * decode_time_per_frame + (
            total_frames * interpolation_overhead
        )

        actual_speedup = original_time / keyframe_time

        return {
            "total_frames": total_frames,
            "keyframe_interval": keyframe_interval,
            "frames_to_decode": frames_to_decode,
            "decode_ratio": decode_ratio,
            "theoretical_speedup": theoretical_speedup,
            "actual_speedup": actual_speedup,
            "time_savings": {
                "original_minutes": original_time / 60,
                "keyframe_minutes": keyframe_time / 60,
                "saved_minutes": (original_time - keyframe_time) / 60,
            },
        }
