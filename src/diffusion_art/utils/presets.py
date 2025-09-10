"""Image preset loading utilities."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


class ImagePresets:
    """Manages loading of preset image pairs for testing interpolation."""

    def __init__(self, assets_dir: str = "assets/test_images"):
        self.assets_dir = Path(assets_dir)

    def get_available_pairs(self) -> List[Dict[str, str]]:
        """Get list of available image pairs.

        Returns:
            List of dicts with 'name', 'category', 'a_path', 'b_path' keys
        """
        pairs: List[dict] = []

        if not self.assets_dir.exists():
            return pairs

        # Find all _A images and look for corresponding _B images
        for a_path in self.assets_dir.glob("*_A.*"):
            # Construct expected B path
            b_name = a_path.name.replace("_A.", "_B.")
            b_path = self.assets_dir / b_name

            if b_path.exists():
                # Parse category and description from filename
                base_name = a_path.stem.replace("_A", "")
                if "_" in base_name:
                    category, description = base_name.split("_", 1)
                else:
                    category, description = "misc", base_name

                pairs.append(
                    {
                        "name": f"{category.title()}: {description.replace('-', ' → ')}",
                        "category": category,
                        "description": description,
                        "a_path": str(a_path),
                        "b_path": str(b_path),
                    }
                )

        # Sort by category then description
        pairs.sort(key=lambda x: (x["category"], x["description"]))
        return pairs

    def load_pair(self, pair_info: Dict[str, str]) -> Tuple[Image.Image, Image.Image]:
        """Load an image pair.

        Args:
            pair_info: Dict from get_available_pairs()

        Returns:
            Tuple of (image_a, image_b)
        """
        img_a = Image.open(pair_info["a_path"])
        img_b = Image.open(pair_info["b_path"])
        return img_a, img_b

    def get_pair_by_name(self, name: str) -> Optional[Tuple[Image.Image, Image.Image]]:
        """Load a pair by its display name.

        Args:
            name: Display name from get_available_pairs()

        Returns:
            Tuple of (image_a, image_b) or None if not found
        """
        pairs = self.get_available_pairs()
        for pair in pairs:
            if pair["name"] == name:
                return self.load_pair(pair)
        return None


# Global instance
presets = ImagePresets()
