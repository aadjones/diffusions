"""Noise injection utilities for latent space experimentation."""

import math
from typing import Callable, Optional, Union

import torch


def add_gaussian_noise(
    latent: torch.Tensor,
    strength: float = 0.1,
    seed: Optional[int] = None,
    adaptive: bool = True,
) -> torch.Tensor:
    """Add Gaussian noise to a latent tensor.

    Args:
        latent: Input latent tensor
        strength: Noise strength (0.0 = no noise, 1.0 = moderate noise)
        seed: Random seed for reproducible noise
        adaptive: Scale noise relative to latent magnitude

    Returns:
        Latent tensor with added noise
    """
    if seed is not None:
        # Set global random seed for reproducible noise
        torch.manual_seed(seed)
        noise = torch.randn_like(latent)
        # Reset to random seed to avoid affecting other operations
        torch.manual_seed(torch.initial_seed())
    else:
        noise = torch.randn_like(latent)

    if adaptive:
        # Scale noise relative to the latent's magnitude for more intuitive control
        latent_mag = latent.std().item()
        scaled_strength = (
            strength * latent_mag * 0.3
        )  # 0.3 makes strength=1.0 reasonable
        return latent + noise * scaled_strength
    else:
        return latent + noise * strength


def breathing_animation(
    base_latent: torch.Tensor,
    frames: int,
    max_strength: float = 0.1,
    pattern: str = "sine",
    seed: Optional[int] = None,
) -> list:
    """Generate breathing animation frames by adding time-varying noise.

    Args:
        base_latent: Base latent tensor to animate
        frames: Number of animation frames
        max_strength: Maximum noise strength
        pattern: Breathing pattern ("sine", "heartbeat", "pulse")
        seed: Random seed for consistent noise pattern

    Returns:
        List of latent tensors for each frame
    """
    # Generate pattern values
    pattern_func = get_pattern_function(pattern)
    pattern_values = [pattern_func(i / frames) for i in range(frames)]

    # Generate consistent noise for all frames
    if seed is not None:
        torch.manual_seed(seed)
        base_noise = torch.randn_like(base_latent)
        torch.manual_seed(torch.initial_seed())
    else:
        base_noise = torch.randn_like(base_latent)

    # Apply pattern to noise strength
    frames_list = []
    for value in pattern_values:
        strength = max_strength * value
        noisy_latent = base_latent + base_noise * strength
        frames_list.append(noisy_latent)

    return frames_list


def get_pattern_function(pattern: str) -> Callable[[float], float]:
    """Get breathing pattern function.

    Args:
        pattern: Pattern name

    Returns:
        Function that takes t (0-1) and returns pattern value (0-1)
    """
    if pattern == "sine":
        return lambda t: abs(math.sin(t * 2 * math.pi))
    elif pattern == "heartbeat":
        # Double pulse pattern like heartbeat
        def heartbeat(t: float) -> float:
            # Create two pulses per cycle
            pulse1 = abs(math.sin(t * 4 * math.pi))
            pulse2 = abs(math.sin(t * 4 * math.pi + math.pi / 2)) * 0.6
            return max(pulse1, pulse2)

        return heartbeat
    elif pattern == "pulse":
        # Sharp pulse pattern
        def pulse(t: float) -> float:
            # Sharp rise and fall
            cycle_pos = (t * 4) % 1
            if cycle_pos < 0.1:
                return cycle_pos * 10  # Quick rise
            elif cycle_pos < 0.3:
                return 1.0 - ((cycle_pos - 0.1) * 5)  # Quick fall
            else:
                return 0.0  # Rest period

        return pulse
    else:
        # Default to sine
        return lambda t: abs(math.sin(t * 2 * math.pi))


def structured_noise(
    latent: torch.Tensor,
    strength: float = 0.1,
    frequency: float = 1.0,
    noise_type: str = "gaussian",
    adaptive: bool = True,
) -> torch.Tensor:
    """Add structured noise patterns to latent tensor.

    Args:
        latent: Input latent tensor
        strength: Noise strength
        frequency: Spatial frequency for structured patterns
        noise_type: Type of noise ("gaussian", "perlin-like", "checkerboard")

    Returns:
        Latent tensor with structured noise
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(latent, strength, adaptive=adaptive)

    elif noise_type == "perlin-like":
        # Simple Perlin-like noise using sinusoids
        b, c, h, w = latent.shape
        device = latent.device

        # Create coordinate grids
        y_coords = torch.linspace(0, frequency * 2 * math.pi, h, device=device)
        x_coords = torch.linspace(0, frequency * 2 * math.pi, w, device=device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Generate smooth noise pattern
        noise_pattern = (
            torch.sin(X) * torch.cos(Y) + torch.sin(X * 2) * torch.cos(Y * 2) * 0.5
        )
        noise_pattern = noise_pattern.unsqueeze(0).unsqueeze(0).expand(b, c, h, w)

        if adaptive:
            latent_mag = latent.std().item()
            scaled_strength = strength * latent_mag * 0.3
            return latent + noise_pattern * scaled_strength
        else:
            return latent + noise_pattern * strength

    elif noise_type == "checkerboard":
        # Checkerboard pattern noise
        b, c, h, w = latent.shape
        device = latent.device

        # Create checkerboard pattern
        checker_h = torch.arange(h, device=device).unsqueeze(1) // int(
            h / (frequency * 4)
        )
        checker_w = torch.arange(w, device=device).unsqueeze(0) // int(
            w / (frequency * 4)
        )
        checker = ((checker_h + checker_w) % 2).float() * 2 - 1  # -1 or 1

        noise_pattern = checker.unsqueeze(0).unsqueeze(0).expand(b, c, h, w)

        if adaptive:
            latent_mag = latent.std().item()
            scaled_strength = strength * latent_mag * 0.3
            return latent + noise_pattern * scaled_strength
        else:
            return latent + noise_pattern * strength

    else:
        return add_gaussian_noise(latent, strength, adaptive=adaptive)
