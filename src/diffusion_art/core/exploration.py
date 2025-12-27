"""Latent space exploration utilities for photography studio."""

from typing import List, Tuple

import torch


def generate_latent_grid(
    center_latent: torch.Tensor,
    grid_size: int = 4,
    exploration_radius: float = 1.0,
    seed: int = 42,
    mode: str = "chaos",
) -> List[torch.Tensor]:
    """Generate a grid of latent variations around a center point.

    Creates a contact sheet of variations by adding different random
    offsets to the center latent. Each variation explores a different
    direction in latent space.

    Args:
        center_latent: Center latent tensor (1, C, H, W)
        grid_size: Number of variations per row/column (total = grid_size²)
        exploration_radius: How far to explore from center (adaptive scaling)
        seed: Random seed for reproducible grids
        mode: Exploration mode ("chaos", "hybrid", "walk")

    Returns:
        List of grid_size² latent tensors
    """
    torch.manual_seed(seed)

    if mode == "chaos":
        # Dramatic random noise - creates wildly different images
        return _generate_chaos_grid(center_latent, grid_size, exploration_radius)
    elif mode == "hybrid":
        # Blend with random latents - creates hybrid images
        return _generate_hybrid_grid(center_latent, grid_size, exploration_radius)
    elif mode == "walk":
        # Random walk from center - accumulating changes
        return _generate_walk_grid(center_latent, grid_size, exploration_radius, seed)
    else:
        # Default to chaos
        return _generate_chaos_grid(center_latent, grid_size, exploration_radius)


def _generate_chaos_grid(
    center_latent: torch.Tensor,
    grid_size: int,
    exploration_radius: float,
) -> List[torch.Tensor]:
    """Pure chaos - massive random noise for dramatic variations."""
    base_magnitude = center_latent.std().item()
    # Much more aggressive scaling - 10x the original conservative factor
    scaled_radius = exploration_radius * base_magnitude * 3.0

    latents = []
    total_variations = grid_size * grid_size

    for i in range(total_variations):
        random_offset = torch.randn_like(center_latent) * scaled_radius
        variation = center_latent + random_offset
        latents.append(variation)

    return latents


def _generate_hybrid_grid(
    center_latent: torch.Tensor,
    grid_size: int,
    exploration_radius: float,
) -> List[torch.Tensor]:
    """Blend center with random latents to create hybrids."""
    base_magnitude = center_latent.std().item()
    latents = []
    total_variations = grid_size * grid_size

    for i in range(total_variations):
        # Generate a completely random latent from natural distribution
        random_latent = torch.randn_like(center_latent) * base_magnitude

        # Blend with center based on exploration radius
        # radius 0.5 = 50% center, 50% random
        # radius 1.0 = 100% random (completely different image)
        blend_factor = min(exploration_radius, 1.0)
        variation = (1 - blend_factor) * center_latent + blend_factor * random_latent
        latents.append(variation)

    return latents


def _generate_walk_grid(
    center_latent: torch.Tensor,
    grid_size: int,
    exploration_radius: float,
    seed: int,
) -> List[torch.Tensor]:
    """Random walk from center - each step accumulates change."""
    from .random_walk import latent_random_walk

    total_variations = grid_size * grid_size
    step_size = 0.1 * exploration_radius

    # Generate a single long walk
    walk_latents = latent_random_walk(
        center_latent,
        steps=total_variations,
        step_size=step_size,
        seed=seed,
    )

    return walk_latents


def generate_pca_grid(
    center_latent: torch.Tensor,
    pca_components: torch.Tensor,
    grid_size: int = 4,
    max_strength: float = 2.0,
) -> List[torch.Tensor]:
    """Generate grid by traversing principal components.

    Creates variations by moving along the most important PCA directions.
    More semantically meaningful than random noise.

    Args:
        center_latent: Center latent tensor (1, C, H, W)
        pca_components: PCA component directions (n_components, latent_dim)
        grid_size: Number of variations per row/column
        max_strength: Maximum traversal distance

    Returns:
        List of grid_size² latent tensors
    """
    latents = []
    n_components = min(grid_size * grid_size, pca_components.shape[0])

    # Flatten center for easier PCA manipulation
    center_flat = center_latent.flatten()

    # Use different PCA components for each cell
    for i in range(grid_size * grid_size):
        if i < n_components:
            # Use a different principal component
            component_idx = i % n_components
            pc_direction = pca_components[component_idx].to(center_latent.device)

            # Vary strength across the grid
            strength = max_strength * ((i % 3) - 1)  # -1, 0, +1 patterns

            # Move along this PC
            varied_flat = center_flat + strength * pc_direction
            varied_latent = varied_flat.view_as(center_latent)
        else:
            # Fallback to center if we run out of components
            varied_latent = center_latent.clone()

        latents.append(varied_latent)

    return latents


def generate_directional_grid(
    center_latent: torch.Tensor,
    grid_size: int = 4,
    exploration_radius: float = 1.0,
) -> List[torch.Tensor]:
    """Generate grid with systematic directional exploration.

    Each grid position represents a specific direction from center,
    arranged in a radial pattern. More organized than random.

    Args:
        center_latent: Center latent tensor
        grid_size: Number of variations per row/column
        exploration_radius: How far to explore from center

    Returns:
        List of grid_size² latent tensors
    """
    import math

    latents = []
    base_magnitude = center_latent.std().item()
    scaled_radius = exploration_radius * base_magnitude * 0.3

    # Center cell is the original
    latents.append(center_latent.clone())

    # Generate variations in a spiral/radial pattern
    angles = torch.linspace(0, 2 * math.pi, grid_size * grid_size - 1)
    distances = torch.linspace(0.3, 1.0, grid_size * grid_size - 1)

    for angle, distance in zip(angles, distances):
        # Create directional offset in first two "conceptual" dimensions
        # (in practice, we create structured noise with this pattern)
        offset = torch.randn_like(center_latent) * scaled_radius * distance

        # Modulate by angle to create directional bias
        angle_factor = math.cos(angle.item())
        offset = offset * (0.5 + 0.5 * angle_factor)

        variation = center_latent + offset
        latents.append(variation)

    return latents


def interpolate_to_latent(
    start_latent: torch.Tensor,
    target_latent: torch.Tensor,
    grid_size: int = 4,
) -> List[torch.Tensor]:
    """Generate grid showing interpolation steps to a target.

    Creates a visual "path" from start to target latent.

    Args:
        start_latent: Starting latent
        target_latent: Target to interpolate toward
        grid_size: Number of steps (grid_size²)

    Returns:
        List of interpolated latents
    """
    from .interpolation import slerp

    latents = []
    total_steps = grid_size * grid_size

    for i in range(total_steps):
        t = i / (total_steps - 1) if total_steps > 1 else 0
        interpolated = slerp(start_latent, target_latent, t)
        latents.append(interpolated)

    return latents


def refine_latent_region(
    interesting_latent: torch.Tensor,
    grid_size: int = 4,
    zoom_factor: float = 0.3,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Zoom into an interesting region with finer variations.

    Like "enhancing" a region - smaller radius, more subtle variations.

    Args:
        interesting_latent: The latent to zoom into
        grid_size: Number of variations
        zoom_factor: How much to zoom (smaller = finer detail)
        seed: Random seed

    Returns:
        List of refined variations
    """
    # Use smaller exploration radius for finer detail
    return generate_latent_grid(
        interesting_latent,
        grid_size=grid_size,
        exploration_radius=zoom_factor,
        seed=seed,
    )
