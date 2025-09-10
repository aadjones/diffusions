"""Gaussian random walk through latent space for continuous exploration."""

from typing import List, Optional, Tuple

import numpy as np
import torch


def distance_threshold_walk(
    start_latent: torch.Tensor,
    steps: int = 50,
    step_size: float = 1.0,
    explore_fraction: float = 0.6,
    seed: Optional[int] = None,
) -> Tuple[List[torch.Tensor], int]:
    """Random walk that explores for a fraction of steps, then SLERPs back home.

    Explores freely for explore_fraction of total steps,
    then SLERPs back home over remaining steps.

    Args:
        start_latent: Starting latent tensor (1, C, H, W)
        steps: Total number of steps in animation
        step_size: Size of each random step during exploration
        explore_fraction: Fraction of steps to spend exploring (0.6 = 60% explore, 40% return)
        seed: Random seed for reproducible walks

    Returns:
        Tuple of (path_latents, turn_around_step)
    """
    if seed is not None:
        torch.manual_seed(seed)

    path = [start_latent.clone()]
    current_latent = start_latent.clone()

    # Adaptive step size based on latent magnitude
    base_magnitude = start_latent.std().item()
    adapted_step_size = step_size * base_magnitude * 0.3

    # Calculate turn around step based on fraction
    turn_around_step = int(steps * explore_fraction)

    # Phase 1: Explore for fixed number of steps
    for step_idx in range(turn_around_step):
        # Take random exploration step
        random_step = torch.randn_like(current_latent) * adapted_step_size
        current_latent = current_latent + random_step
        path.append(current_latent.clone())

    # Phase 2: SLERP back to start over remaining steps
    remaining_steps = steps - turn_around_step
    if remaining_steps > 0:
        from ..core.interpolation import slerp

        furthest_point = current_latent.clone()

        for i in range(1, remaining_steps + 1):
            t = i / remaining_steps  # 0 to 1 over remaining steps
            return_latent = slerp(furthest_point, start_latent, t)
            path.append(return_latent)

    return path, turn_around_step


def latent_random_walk(
    start_latent: torch.Tensor,
    steps: int = 50,
    step_size: float = 1.0,
    return_home: bool = False,
    return_strength: float = 0.05,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Generate a random walk through latent space.

    Args:
        start_latent: Starting latent tensor (1, C, H, W)
        steps: Number of walk steps
        step_size: Size of each random step (larger = more dramatic changes)
        return_home: Whether to gradually bias walk back toward start
        return_strength: How strongly to pull back toward start (if return_home=True)
        seed: Random seed for reproducible walks

    Returns:
        List of latent tensors along the walk path
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize walk path
    path = [start_latent.clone()]
    current_latent = start_latent.clone()

    # Adaptive step size based on latent magnitude
    base_magnitude = start_latent.std().item()
    adapted_step_size = (
        step_size * base_magnitude * 0.3
    )  # Similar to adaptive noise scaling

    for step_idx in range(steps):
        # Generate random step
        random_step = torch.randn_like(current_latent) * adapted_step_size

        # Apply return-home bias if enabled
        if return_home and step_idx > 0:
            # Vector pointing back toward start
            home_vector = start_latent - current_latent
            # Add small bias toward home
            random_step += home_vector * return_strength

        # Take the step
        current_latent = current_latent + random_step
        path.append(current_latent.clone())

    return path


def biased_random_walk(
    start_latent: torch.Tensor,
    target_latent: torch.Tensor,
    steps: int = 50,
    step_size: float = 1.0,
    bias_strength: float = 0.1,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Generate a biased random walk toward a target latent.

    Args:
        start_latent: Starting latent tensor
        target_latent: Target to bias walk toward
        steps: Number of walk steps
        step_size: Size of each random step
        bias_strength: How strongly to bias toward target (0=pure random, 1=direct path)
        seed: Random seed

    Returns:
        List of latent tensors along the biased walk
    """
    if seed is not None:
        torch.manual_seed(seed)

    path = [start_latent.clone()]
    current_latent = start_latent.clone()

    # Adaptive step size
    base_magnitude = start_latent.std().item()
    adapted_step_size = step_size * base_magnitude * 0.3

    for step_idx in range(steps):
        # Generate random step
        random_step = torch.randn_like(current_latent) * adapted_step_size

        # Add bias toward target
        if bias_strength > 0:
            target_direction = target_latent - current_latent
            target_direction_normalized = target_direction / (
                target_direction.norm() + 1e-8
            )
            bias_step = target_direction_normalized * adapted_step_size * bias_strength
            random_step += bias_step

        # Take the step
        current_latent = current_latent + random_step
        path.append(current_latent.clone())

    return path


def circular_random_walk(
    start_latent: torch.Tensor,
    steps: int = 50,
    radius: float = 2.0,
    chaos: float = 0.5,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Generate a roughly circular random walk that orbits around the start point.

    This creates a path that moves in a circle through latent space,
    never returning to the original position (unlike breathing patterns).

    Args:
        start_latent: Center point of the circular orbit
        steps: Number of walk steps
        radius: Target radius from center
        chaos: Amount of randomness (0=perfect circle, 1=chaotic)
        seed: Random seed

    Returns:
        List of latent tensors along the circular walk
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Start at the center
    path = [start_latent.clone()]

    # Adaptive radius based on latent magnitude
    base_magnitude = start_latent.std().item()
    adapted_radius = radius * base_magnitude * 0.3

    # Pick consistent dimensions for the circular plane (don't change per step!)
    flat_start = start_latent.flatten()
    n_dims = flat_start.shape[0]

    # Use first two dimensions for primary circular motion, but could randomize once
    torch.manual_seed(seed or 42)  # Ensure consistent plane selection
    dim1 = torch.randint(0, n_dims // 4, (1,)).item()  # First quarter of dimensions
    dim2 = torch.randint(n_dims // 4, n_dims // 2, (1,)).item()  # Second quarter

    # Move to starting position on circle (offset from center)
    initial_offset = torch.zeros_like(flat_start)
    initial_offset[dim1] = adapted_radius  # Start at angle 0
    current_position = flat_start + initial_offset
    path.append(current_position.view_as(start_latent))

    for step_idx in range(1, steps):
        # Circular component - complete circle over all steps
        angle = (step_idx / steps) * 2 * np.pi

        # Create circular displacement from center
        circular_displacement = torch.zeros_like(flat_start)
        circular_displacement[dim1] = torch.cos(torch.tensor(angle)) * adapted_radius
        circular_displacement[dim2] = torch.sin(torch.tensor(angle)) * adapted_radius

        # Position on circle relative to center
        base_position = flat_start + circular_displacement

        # Add chaos (random walk component)
        if chaos > 0:
            # Smaller random steps that don't destroy the circular pattern
            random_drift = torch.randn_like(flat_start) * adapted_radius * chaos * 0.3
            base_position += random_drift

        # Store new position and reshape
        new_latent = base_position.view_as(start_latent)
        path.append(new_latent)

    return path


def momentum_random_walk(
    start_latent: torch.Tensor,
    steps: int = 50,
    step_size: float = 1.0,
    momentum: float = 0.7,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Random walk with momentum - steps influenced by previous direction.

    Args:
        start_latent: Starting latent tensor
        steps: Number of walk steps
        step_size: Size of each step
        momentum: How much previous step influences current step (0-1)
        seed: Random seed

    Returns:
        List of latent tensors along the momentum walk
    """
    if seed is not None:
        torch.manual_seed(seed)

    path = [start_latent.clone()]
    current_latent = start_latent.clone()

    # Adaptive step size
    base_magnitude = start_latent.std().item()
    adapted_step_size = step_size * base_magnitude * 0.3

    # Initialize momentum vector
    momentum_vector = torch.zeros_like(current_latent)

    for step_idx in range(steps):
        # Generate new random step
        random_step = torch.randn_like(current_latent) * adapted_step_size

        # Combine with momentum from previous step
        combined_step = momentum * momentum_vector + (1 - momentum) * random_step

        # Update momentum for next step
        momentum_vector = combined_step

        # Take the step
        current_latent = current_latent + combined_step
        path.append(current_latent.clone())

    return path
