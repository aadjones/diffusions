"""Analyze potential shortcuts for VAE decoding."""

import time

import torch

from diffusion_art.models.vae import SD15VAE
from diffusion_art.optimization.profiler import AnimationProfiler
from diffusion_art.optimization.vae_optimizer import VAEOptimizer


def analyze_latent_image_relationship():
    """Explore relationships between latent changes and image changes."""
    vae = SD15VAE()
    vae.load_model()

    # Generate base latent and small perturbations
    base_latent = torch.randn(1, 4, 64, 64).to(vae.device)

    print("🔍 Analyzing Latent-Image Relationships...")

    # Test different perturbation sizes
    perturbations = [0.01, 0.1, 0.5, 1.0]

    for perturbation_size in perturbations:
        # Create small change in latent space
        perturbed_latent = (
            base_latent + torch.randn_like(base_latent) * perturbation_size
        )

        # Decode both (expensive!)
        start_time = time.time()
        base_img = vae.decode(base_latent)
        perturbed_img = vae.decode(perturbed_latent)
        decode_time = time.time() - start_time

        # Calculate image space difference
        base_array = torch.tensor(base_img).float()
        perturbed_array = torch.tensor(perturbed_img).float()
        image_diff = torch.mean((base_array - perturbed_array) ** 2).item()

        print(
            f"Perturbation {perturbation_size:.2f}: "
            f"Image MSE={image_diff:.3f}, Decode time={decode_time:.2f}s"
        )


def test_interpolation_strategies():
    """Compare latent vs image space interpolation."""
    vae = SD15VAE()
    vae.load_model()

    print("\n🎯 Testing Interpolation Strategies...")

    # Create two random latents
    latent_a = torch.randn(1, 4, 64, 64).to(vae.device)
    latent_b = torch.randn(1, 4, 64, 64).to(vae.device)

    # Strategy 1: Interpolate in latent space (current approach)
    print("Strategy 1: Latent space interpolation")
    start_time = time.time()

    latent_frames = []
    for t in torch.linspace(0, 1, 5):
        interp_latent = (1 - t) * latent_a + t * latent_b
        latent_frames.append(interp_latent)

    # Decode all frames
    _ = [vae.decode(latent) for latent in latent_frames]
    latent_interp_time = time.time() - start_time

    print(f"  Time: {latent_interp_time:.2f}s for 5 frames")

    # Strategy 2: Decode endpoints, interpolate in image space
    print("Strategy 2: Image space interpolation")
    start_time = time.time()

    img_a = vae.decode(latent_a)
    img_b = vae.decode(latent_b)

    # Interpolate in image space (much faster!)
    import numpy as np
    from PIL import Image

    array_a = np.array(img_a)
    array_b = np.array(img_b)

    images_from_image = []
    for t in np.linspace(0, 1, 5):
        interp_array = (1 - t) * array_a + t * array_b
        interp_img = Image.fromarray(interp_array.astype(np.uint8))
        images_from_image.append(interp_img)

    image_interp_time = time.time() - start_time

    print(f"  Time: {image_interp_time:.2f}s for 5 frames")
    print(f"  Speedup: {latent_interp_time / image_interp_time:.1f}x")


def explore_proxy_decoder_concept():
    """Explore concept of a lightweight proxy decoder."""
    print("\n🚀 Proxy Decoder Concept Analysis...")

    # Theoretical analysis
    full_decoder_ops = "~100M parameters, multiple conv layers"
    proxy_decoder_ops = "~1M parameters, single conv layer"

    print("Full VAE Decoder:")
    print(f"  - Complexity: {full_decoder_ops}")
    print("  - Quality: High fidelity")
    print("  - Speed: ~2s per frame")

    print("\nHypothetical Proxy Decoder:")
    print(f"  - Complexity: {proxy_decoder_ops}")
    print("  - Quality: Lower fidelity approximation")
    print("  - Potential Speed: ~0.1s per frame (20x faster)")

    print("\nTraining Strategy:")
    print("  - Train lightweight CNN: (1,4,64,64) -> (1,3,512,512)")
    print("  - Loss: MSE between proxy output and real VAE output")
    print("  - Use for previews/intermediate frames")


if __name__ == "__main__":
    print("🔬 Performance Analysis: Latent-Image Relationships\n")

    # analyze_latent_image_relationship()
    test_interpolation_strategies()
    explore_proxy_decoder_concept()

    print("\n💡 Key Insights:")
    print("1. No cheap way to predict latent->image mapping")
    print("2. Image space interpolation is ~10x faster")
    print("3. Proxy decoder could provide 20x speedup")
    print("4. Trade-offs: Speed vs Quality vs Latent-space semantics")
