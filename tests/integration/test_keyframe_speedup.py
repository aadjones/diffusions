"""Test keyframe rendering speedup."""

import torch

from diffusion_art.models.vae import SD15VAE
from diffusion_art.optimization.keyframe_renderer import KeyframeRenderer


def test_keyframe_speedup():
    """Test keyframe rendering approach."""
    print("🚀 Testing Keyframe Rendering Speedup\n")

    # Setup
    vae = SD15VAE()
    vae.load_model()
    renderer = KeyframeRenderer(vae)

    # Create a test animation sequence
    print("🎬 Creating test latent sequence...")
    base_latent = torch.randn(1, 4, 64, 64).to(vae.device)

    # Simple linear interpolation in latent space for testing
    target_latent = torch.randn(1, 4, 64, 64).to(vae.device)

    latent_sequence = []
    for i in range(24):  # 24 frame test sequence
        alpha = i / 23
        interp_latent = (1 - alpha) * base_latent + alpha * target_latent
        latent_sequence.append(interp_latent)

    print(f"📝 Created {len(latent_sequence)} frame sequence")

    # Test different keyframe intervals
    print("\n📊 Testing Quality vs Speed Tradeoffs:")
    results = renderer.analyze_quality_vs_speed_tradeoff(
        latent_sequence, test_intervals=[2, 4, 8, 12]
    )

    print("\n📈 Results Summary:")
    print("Interval | Speedup | Quality (MSE) | Frames Decoded")
    print("-" * 50)
    for interval, data in results.items():
        print(
            f"{interval:8d} | {data['speedup']:6.2f}x | {data['quality_mse']:12.1f} | {data['frames_decoded']:3d}/24"
        )

    # Estimate production scaling
    print("\n🎯 Production Estimates:")
    for frames in [100, 500, 1000]:
        estimate = renderer.estimate_production_speedup(frames, keyframe_interval=8)
        print(f"\n{frames} frame animation (keyframe every 8):")
        print(f"  Speedup: {estimate['actual_speedup']:.1f}x")
        print(f"  Time saved: {estimate['time_savings']['saved_minutes']:.1f} minutes")
        print(f"  Decode only: {estimate['frames_to_decode']}/{frames} frames")


if __name__ == "__main__":
    test_keyframe_speedup()
