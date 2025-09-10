"""PCA-based latent space traversal for meaningful breathing effects."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA


class LatentPCATraversal:
    """Compute and traverse principal components in latent space."""

    def __init__(self, n_components: int = 50):
        """Initialize PCA traversal.

        Args:
            n_components: Number of principal components to compute
        """
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.latent_shape: Optional[Tuple[int, ...]] = None
        self.mean_latent: Optional[torch.Tensor] = None

    def fit(self, latent_samples: torch.Tensor) -> None:
        """Fit PCA to a collection of latent samples.

        Args:
            latent_samples: Tensor of shape (n_samples, channels, height, width)
        """
        # Store original shape for reconstruction
        self.latent_shape = latent_samples.shape[1:]

        # Flatten latents for PCA
        n_samples = latent_samples.shape[0]
        flattened = latent_samples.reshape(n_samples, -1).cpu().numpy()

        # Compute mean for centering
        self.mean_latent = torch.from_numpy(np.mean(flattened, axis=0)).float()

        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(flattened)

    def generate_samples_from_base(
        self,
        base_latent: torch.Tensor,
        n_samples: int = 100,
        variation_strength: float = 1.5,
    ) -> torch.Tensor:
        """Generate meaningful latent samples for PCA fitting.

        Instead of random Gaussian noise, generate structured variations
        that are more likely to correspond to meaningful semantic changes.

        Args:
            base_latent: Base latent tensor (1, C, H, W)
            n_samples: Number of samples to generate
            variation_strength: Strength of structured variations

        Returns:
            Tensor of latent samples (n_samples, C, H, W)
        """
        samples = []

        # Method 1: Channel-wise variation (each of the 4 VAE channels has meaning)
        for i in range(n_samples // 3):
            sample = base_latent.clone()

            # Vary each channel independently with different strengths
            for c in range(4):  # 4 channels in SD15 latent
                channel_variation = (
                    torch.randn_like(sample[0, c])
                    * variation_strength
                    * (0.5 + c * 0.3)
                )
                sample[0, c] += channel_variation

            samples.append(sample)

        # Method 2: Spatial frequency variations (low vs high frequency changes)
        for i in range(n_samples // 3):
            sample = base_latent.clone()

            # Low frequency changes (global structure)
            low_freq = (
                torch.randn(1, 4, 8, 8, device=base_latent.device)
                * variation_strength
                * 2.0
            )
            low_freq = torch.nn.functional.interpolate(
                low_freq, size=(64, 64), mode="bilinear", align_corners=False
            )
            sample += low_freq

            samples.append(sample)

        # Method 3: Directional variations (create systematic directional changes)
        remaining = n_samples - len(samples)
        for i in range(remaining):
            sample = base_latent.clone()

            # Create gradient-like variations across spatial dimensions
            h_gradient = (
                torch.linspace(-1, 1, 64, device=base_latent.device)
                .view(1, 64)
                .expand(64, 64)
                * variation_strength
            )
            v_gradient = (
                torch.linspace(-1, 1, 64, device=base_latent.device)
                .view(64, 1)
                .expand(64, 64)
                * variation_strength
            )

            # Apply to random channels
            active_channels = torch.randperm(4, device=base_latent.device)[
                :2
            ]  # Pick 2 channels
            for c in active_channels:
                if i % 2 == 0:
                    sample[0, c] += h_gradient
                else:
                    sample[0, c] += v_gradient

            samples.append(sample)

        return torch.cat(samples, dim=0)

    def traverse_component(
        self,
        base_latent: torch.Tensor,
        component_idx: int,
        strength: float,
        n_steps: int = 30,
    ) -> List[torch.Tensor]:
        """Traverse along a specific principal component.

        Args:
            base_latent: Base latent tensor to start from
            component_idx: Which PC to traverse (0 = most variance)
            strength: How far to traverse (-strength to +strength)
            n_steps: Number of steps in the traversal

        Returns:
            List of latent tensors along the traversal
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted first with fit()")

        # Get the principal component direction and move to same device as base_latent
        pc_direction = (
            torch.from_numpy(self.pca.components_[component_idx])
            .float()
            .to(base_latent.device)
        )

        # Create traversal steps
        step_values = torch.linspace(-strength, strength, n_steps)
        traversal_latents = []

        base_flat = base_latent.flatten()

        for step in step_values:
            # Move along PC direction
            traversed_flat = base_flat + step * pc_direction
            traversed_latent = traversed_flat.view_as(base_latent)
            traversal_latents.append(traversed_latent)

        return traversal_latents

    def breathing_along_component(
        self,
        base_latent: torch.Tensor,
        component_idx: int,
        max_strength: float = 2.0,
        frames: int = 30,
        pattern: str = "sine",
    ) -> List[torch.Tensor]:
        """Create breathing animation along a principal component.

        Args:
            base_latent: Base latent to animate
            component_idx: Which PC to use for breathing
            max_strength: Maximum traversal strength
            frames: Number of animation frames
            pattern: Breathing pattern ("sine", "heartbeat", "pulse")

        Returns:
            List of latent frames for breathing animation
        """
        from .noise import get_pattern_function

        if self.pca is None:
            raise ValueError("PCA must be fitted first with fit()")

        # Get pattern function
        pattern_func = get_pattern_function(pattern)

        # Get PC direction and move to same device as base_latent
        pc_direction = (
            torch.from_numpy(self.pca.components_[component_idx])
            .float()
            .to(base_latent.device)
        )
        base_flat = base_latent.flatten()

        breathing_frames = []

        # Effective multiplier to make changes visible (was the key missing piece)
        effective_multiplier = 15.0  # Scale up the traversal for visible changes

        for i in range(frames):
            t = i / frames
            # Map pattern to -max_strength to +max_strength range
            raw_strength = (pattern_func(t) * 2 - 1) * max_strength
            # Apply effective multiplier for meaningful latent space changes
            effective_strength = raw_strength * effective_multiplier

            # Move along PC direction
            breathed_flat = base_flat + effective_strength * pc_direction
            breathed_latent = breathed_flat.view_as(base_latent)
            breathing_frames.append(breathed_latent)

        return breathing_frames

    def get_component_info(self) -> Optional[np.ndarray]:
        """Get explained variance ratio for each component.

        Returns:
            Array of explained variance ratios, or None if not fitted
        """
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_

    def get_top_components(self, n_top: int = 10) -> List[Tuple[int, float]]:
        """Get top N components by explained variance.

        Args:
            n_top: Number of top components to return

        Returns:
            List of (component_index, explained_variance_ratio) tuples
        """
        if self.pca is None:
            return []

        ratios = self.pca.explained_variance_ratio_
        indexed_ratios = [(i, ratio) for i, ratio in enumerate(ratios)]
        return sorted(indexed_ratios, key=lambda x: x[1], reverse=True)[:n_top]


def quick_pca_breathing(
    base_latent: torch.Tensor,
    component_idx: int = 0,
    max_strength: float = 2.0,
    frames: int = 30,
    pattern: str = "sine",
    n_samples: int = 50,
) -> Tuple[List[torch.Tensor], LatentPCATraversal]:
    """Quick helper to generate PCA breathing without manual setup.

    Args:
        base_latent: Base latent tensor
        component_idx: Which principal component to use
        max_strength: Maximum breathing strength
        frames: Number of animation frames
        pattern: Breathing pattern
        n_samples: Number of samples for PCA fitting

    Returns:
        Tuple of (breathing_frames, pca_traversal_object)
    """
    # Create PCA traversal object
    pca_trav = LatentPCATraversal(n_components=min(20, n_samples - 1))

    # Generate samples around base latent
    samples = pca_trav.generate_samples_from_base(base_latent, n_samples=n_samples)

    # Fit PCA
    pca_trav.fit(samples)

    # Generate breathing animation
    breathing_frames = pca_trav.breathing_along_component(
        base_latent, component_idx, max_strength, frames, pattern
    )

    return breathing_frames, pca_trav
