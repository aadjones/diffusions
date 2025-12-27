# Exploration API Reference

This document shows how to use the Photography Studio's exploration functions programmatically.

## Basic Usage

```python
from src.diffusion_art.models.vae import SD15VAE
from src.diffusion_art.core.exploration import generate_latent_grid
from PIL import Image

# Initialize VAE
vae = SD15VAE()
vae.load_model()

# Load and encode starting image
img = Image.open("portrait.jpg")
center_latent = vae.encode(img)

# Generate 16 variations (4×4 grid)
variations = generate_latent_grid(
    center_latent,
    grid_size=4,
    exploration_radius=1.0,
    seed=42
)

# Decode all variations
images = [vae.decode(latent) for latent in variations]

# Save contact sheet
for i, img in enumerate(images):
    img.save(f"variation_{i:02d}.png")
```

## Exploration Functions

### `generate_latent_grid()`

Generate random variations around a center point.

```python
from src.diffusion_art.core.exploration import generate_latent_grid

latents = generate_latent_grid(
    center_latent=base_latent,    # Starting point
    grid_size=4,                   # 4×4 = 16 variations
    exploration_radius=1.0,        # How far to explore
    seed=42                        # Reproducible results
)
# Returns: List[torch.Tensor] of length grid_size²
```

**Parameters:**
- `center_latent`: The base latent tensor (1, 4, 64, 64)
- `grid_size`: Grid dimensions (3, 4, or 5 recommended)
- `exploration_radius`: Distance from center (0.1-3.0)
  - 0.1-0.5: Subtle variations
  - 1.0: Balanced exploration
  - 2.0-3.0: Dramatic changes
- `seed`: Random seed for reproducibility

### `generate_directional_grid()`

Generate variations in a systematic radial pattern.

```python
from src.diffusion_art.core.exploration import generate_directional_grid

latents = generate_directional_grid(
    center_latent=base_latent,
    grid_size=4,
    exploration_radius=1.0
)
```

**Use cases:**
- More organized coverage than random
- Systematic exploration of all "directions"
- Good for creating comparison grids

### `refine_latent_region()`

Zoom into a region with finer-grained variations.

```python
from src.diffusion_art.core.exploration import refine_latent_region

# Found interesting variation, now refine it
refined_latents = refine_latent_region(
    interesting_latent=variations[10],  # The one you liked
    grid_size=4,
    zoom_factor=0.3,  # Smaller = finer detail
    seed=42
)
```

**Parameters:**
- `zoom_factor`: How tight to zoom (0.1-1.0)
  - 0.1: Very subtle differences
  - 0.3: Good balance (default)
  - 1.0: Same as normal exploration

### `interpolate_to_latent()`

Create a path from one latent to another.

```python
from src.diffusion_art.core.exploration import interpolate_to_latent

# Show progression from A to B
path = interpolate_to_latent(
    start_latent=latent_a,
    target_latent=latent_b,
    grid_size=4  # 16 steps from A to B
)
```

## Complete Example: Multi-Level Exploration

```python
from src.diffusion_art.models.vae import SD15VAE
from src.diffusion_art.core.exploration import (
    generate_latent_grid,
    refine_latent_region
)
from PIL import Image
import os

# Setup
vae = SD15VAE()
vae.load_model()
img = Image.open("input.jpg")
base_latent = vae.encode(img)

# Level 1: Broad search
print("Level 1: Exploring 16 variations...")
level1_latents = generate_latent_grid(
    base_latent,
    grid_size=4,
    exploration_radius=1.5,
    seed=42
)

# Decode level 1
level1_images = [vae.decode(lat) for lat in level1_latents]

# Save level 1
os.makedirs("exploration/level1", exist_ok=True)
for i, img in enumerate(level1_images):
    img.save(f"exploration/level1/var_{i:02d}.png")

# Manually pick interesting one (or use automated metric)
interesting_idx = 7  # Example: variation #7 was interesting

# Level 2: Refine the interesting region
print("Level 2: Refining variation #7...")
level2_latents = refine_latent_region(
    level1_latents[interesting_idx],
    grid_size=4,
    zoom_factor=0.3,
    seed=43
)

# Decode level 2
level2_images = [vae.decode(lat) for lat in level2_latents]

# Save level 2
os.makedirs("exploration/level2", exist_ok=True)
for i, img in enumerate(level2_images):
    img.save(f"exploration/level2/var_{i:02d}.png")

print("Exploration complete! Check exploration/ directory")
```

## Batch Processing

For efficiency, process latents in batches:

```python
def batch_decode(vae, latents, batch_size=4):
    """Decode latents in batches."""
    images = []
    for i in range(0, len(latents), batch_size):
        batch = latents[i:i + batch_size]
        for latent in batch:
            img = vae.decode(latent)
            images.append(img)
    return images

# Use it
images = batch_decode(vae, variations, batch_size=4)
```

## Creating Contact Sheet Images

Combine individual images into a grid:

```python
from PIL import Image

def create_contact_sheet(images, grid_size, image_size=256):
    """Combine images into a contact sheet."""
    sheet_width = grid_size * image_size
    sheet_height = grid_size * image_size

    contact_sheet = Image.new('RGB', (sheet_width, sheet_height))

    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size

        # Resize image
        img_resized = img.resize((image_size, image_size))

        # Paste into contact sheet
        x = col * image_size
        y = row * image_size
        contact_sheet.paste(img_resized, (x, y))

    return contact_sheet

# Use it
images = [vae.decode(lat) for lat in variations]
sheet = create_contact_sheet(images, grid_size=4)
sheet.save("contact_sheet.png")
```

## Automated Exploration

Explore automatically and save interesting results:

```python
import torch

def explore_automatically(vae, base_latent, depth=3, grid_size=4):
    """Automatically explore latent space to specified depth."""

    def interesting_score(latent):
        """Simple metric: variance in latent space."""
        return latent.std().item()

    current_latent = base_latent
    path = [current_latent]

    for level in range(depth):
        print(f"Exploring level {level + 1}...")

        # Generate variations
        variations = generate_latent_grid(
            current_latent,
            grid_size=grid_size,
            exploration_radius=1.0,
            seed=42 + level
        )

        # Score each variation
        scores = [interesting_score(lat) for lat in variations]

        # Pick most interesting
        best_idx = scores.index(max(scores))
        current_latent = variations[best_idx]
        path.append(current_latent)

        print(f"  Selected variation #{best_idx} (score: {scores[best_idx]:.3f})")

    return path

# Use it
exploration_path = explore_automatically(vae, base_latent, depth=5)

# Decode and save path
for i, latent in enumerate(exploration_path):
    img = vae.decode(latent)
    img.save(f"auto_explore_step_{i}.png")
```

## Integration with Existing Code

Works seamlessly with other Diffusion Art features:

```python
from src.diffusion_art.core.interpolation import slerp
from src.diffusion_art.core.random_walk import latent_random_walk
from src.diffusion_art.core.exploration import generate_latent_grid

# Explore around an interpolated point
interp = slerp(latent_a, latent_b, t=0.5)
variations = generate_latent_grid(interp, grid_size=4)

# Explore around a random walk position
walk = latent_random_walk(base_latent, steps=50)
interesting_point = walk[25]  # Midpoint
variations = generate_latent_grid(interesting_point, grid_size=4)
```

## Performance Considerations

**Memory management:**
```python
import torch

# Clear GPU cache between batches
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Use smaller grids for faster iteration
quick_explore = generate_latent_grid(lat, grid_size=3)  # 9 images
thorough_explore = generate_latent_grid(lat, grid_size=5)  # 25 images
```

**Decoding time estimates:**
- 3×3 grid = 9 images ≈ 18 seconds
- 4×4 grid = 16 images ≈ 32 seconds
- 5×5 grid = 25 images ≈ 50 seconds

(Based on ~2 seconds per decode on Apple Silicon MPS)
