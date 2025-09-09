"""Feature flags for enabling/disabling experimental features."""

import os
from typing import Dict


class FeatureFlags:
    """Centralized feature flag management."""

    def __init__(self):
        # Default feature states
        self._flags: Dict[str, bool] = {
            # Debug and development features
            "debug_panel": self._get_env_bool("DEBUG_PANEL", default=False),
            "latent_metrics": self._get_env_bool("LATENT_METRICS", default=False),
            "comparison_mode": self._get_env_bool("COMPARISON_MODE", default=True),
            # UI features
            "animation_export": self._get_env_bool("ANIMATION_EXPORT", default=True),
            "multi_interpolation": self._get_env_bool(
                "MULTI_INTERPOLATION", default=False
            ),
            # Experimental features
            "atlas_view": self._get_env_bool("ATLAS_VIEW", default=False),
            "audio_sync": self._get_env_bool("AUDIO_SYNC", default=False),
            "fisher_curvature": self._get_env_bool("FISHER_CURVATURE", default=False),
        }

    def _get_env_bool(self, env_var: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(env_var, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            return default

    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self._flags.get(feature, False)

    def enable(self, feature: str) -> None:
        """Enable a feature at runtime."""
        self._flags[feature] = True

    def disable(self, feature: str) -> None:
        """Disable a feature at runtime."""
        self._flags[feature] = False

    def get_all(self) -> Dict[str, bool]:
        """Get all feature flags."""
        return self._flags.copy()

    def toggle(self, feature: str) -> bool:
        """Toggle a feature and return new state."""
        self._flags[feature] = not self._flags.get(feature, False)
        return self._flags[feature]


# Global feature flags instance
features = FeatureFlags()
