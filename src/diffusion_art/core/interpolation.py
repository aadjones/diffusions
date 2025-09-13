"""Latent space interpolation functions."""

from typing import List

import torch


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent tensors.

    SLERP maintains constant magnitude while interpolating along the great circle
    path on the unit sphere. This often produces smoother transitions than linear
    interpolation (LERP) in high-dimensional latent spaces.

    Args:
        z0: Starting latent tensor
        z1: Ending latent tensor
        t: Interpolation parameter (0.0 = z0, 1.0 = z1)

    Returns:
        Interpolated latent tensor
    """
    # Flatten to vectors for easier computation
    a, b = z0.flatten(1), z1.flatten(1)

    # Normalize to unit vectors
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)

    # Compute angle between vectors (clamped to avoid numerical issues)
    dot = (a * b).sum(dim=1, keepdim=True).clamp(-0.999999, 0.999999)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Compute interpolation weights
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta

    # Interpolate and restore original shape
    out = (w0 * a + w1 * b).view_as(z0)

    # Restore original magnitude
    original_mag = z0.flatten(1).norm(dim=1, keepdim=True)
    out = out.flatten(1)
    out = out / (out.norm(dim=1, keepdim=True) + 1e-8) * original_mag

    return out.view_as(z0)


def lerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two latent tensors.

    Simple linear interpolation that blends directly in latent space.

    Args:
        z0: Starting latent tensor
        z1: Ending latent tensor
        t: Interpolation parameter (0.0 = z0, 1.0 = z1)

    Returns:
        Interpolated latent tensor
    """
    return (1 - t) * z0 + t * z1


def multi_slerp(latents: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """Multi-way spherical interpolation between multiple latent tensors.

    Args:
        latents: List of latent tensors to interpolate between
        weights: List of weights (should sum to 1.0)

    Returns:
        Interpolated latent tensor
    """
    if len(latents) != len(weights):
        raise ValueError("Number of latents must match number of weights")

    if len(latents) < 2:
        raise ValueError("Need at least 2 latents for interpolation")

    # Normalize weights to sum to 1
    weights_tensor = torch.tensor(weights)
    weights_tensor = weights_tensor / weights_tensor.sum()

    # Start with first latent
    result = weights_tensor[0] * latents[0]

    # Progressively slerp with remaining latents
    for i in range(1, len(latents)):
        # Weight of current result vs new latent
        total_weight = weights_tensor[: i + 1].sum()
        t = weights_tensor[i] / total_weight if total_weight > 0 else 0
        result = slerp(result, latents[i], float(t))

    return result


def create_interpolation_path(
    z0: torch.Tensor, z1: torch.Tensor, steps: int, method: str = "slerp"
) -> List[torch.Tensor]:
    """Create a path of interpolated latents between two endpoints.

    Args:
        z0: Starting latent tensor
        z1: Ending latent tensor
        steps: Number of interpolation steps (including endpoints)
        method: Interpolation method ("slerp" or "lerp")

    Returns:
        List of interpolated latent tensors
    """
    if steps < 2:
        raise ValueError("Need at least 2 steps for interpolation path")

    t_values = torch.linspace(0, 1, steps, device=z0.device)
    interpolate_fn = slerp if method == "slerp" else lerp

    return [interpolate_fn(z0, z1, float(t)) for t in t_values]


def slerp_batch(
    z0: torch.Tensor, z1: torch.Tensor, t_values: torch.Tensor
) -> List[torch.Tensor]:
    """Vectorized SLERP for multiple t values at once.

    More efficient than calling slerp() in a loop when generating many interpolation steps.

    Args:
        z0: Starting latent tensor (1, C, H, W)
        z1: Ending latent tensor (1, C, H, W)
        t_values: Tensor of interpolation values (N,)

    Returns:
        List of interpolated tensors
    """
    # Flatten for easier vectorized computation
    a, b = z0.flatten(1), z1.flatten(1)

    # Normalize to unit vectors
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)

    # Compute angle between vectors (only once)
    dot = (a * b).sum(dim=1, keepdim=True).clamp(-0.999999, 0.999999)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Compute original magnitude for restoration
    original_mag = z0.flatten(1).norm(dim=1, keepdim=True)

    results = []
    for t in t_values:
        # Compute interpolation weights for this t
        w0 = torch.sin((1 - t) * theta) / sin_theta
        w1 = torch.sin(t * theta) / sin_theta

        # Interpolate and restore shape/magnitude
        out = (w0 * a + w1 * b).view_as(z0)
        out = out.flatten(1)
        out = out / (out.norm(dim=1, keepdim=True) + 1e-8) * original_mag
        results.append(out.view_as(z0))

    return results
