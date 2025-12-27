# 📷 What's New: Photography Studio

## Overview

A new exploration mode that turns latent space into a **photography contact sheet** experience. Instead of watching slow animations, you generate grids of variations and interactively "hunt" for interesting images.

## How It Works

```
1. Upload image → 2. Generate grid → 3. Click interesting one → 4. Repeat
```

Each click takes you **deeper into latent space**, following threads of aesthetic interest.

## Key Features

### ✅ Interactive Contact Sheets
- Generate 3×3, 4×4, or 5×5 grids of variations
- Each variation explores a different "direction" from your current point
- Click any variation to make it the new center point

### ✅ Smart Exploration Modes
- **Random Variations**: Chaotic exploration (good for broad search)
- **Directional**: Systematic radial coverage
- **Refined Zoom**: Subtle variations for fine-tuning

### ✅ Navigation & History
- **Back button**: Return to previous exploration points
- **Reset**: Jump back to original image
- **Random seed**: Get different variations from same location
- **Depth tracking**: Know how deep you've explored

### ✅ Performance Optimized
- Grid sizes tuned for practical use (20-55 seconds per sheet)
- Progress indicators during generation
- Sparse sampling instead of dense animation

## Why This Is Better Than Animation

| Feature | Animation | Contact Sheet |
|---------|-----------|---------------|
| **Time to see results** | Minutes | Seconds |
| **User agency** | None (watch passively) | High (click to explore) |
| **Discoveries** | Linear path | Branching exploration |
| **Output** | Video file | Collection of images |
| **Iteration speed** | Slow | Fast |

## Example Workflows

### Workflow 1: Finding the Perfect Variation
```
1. Upload portrait
2. Generate 4×4 grid (radius 1.5)
3. Click dreamlike variation
4. Switch to "Refined Zoom" (radius 0.3)
5. Generate 3×3 grid
6. Export the perfect subtle variation
```

### Workflow 2: Systematic Exploration
```
1. Upload landscape
2. Use "Directional" mode
3. Generate 5×5 grid
4. Click top-right quadrant
5. Repeat for full coverage
6. Build aesthetic map of latent space
```

### Workflow 3: Following Threads
```
1. Upload abstract art
2. Quick 3×3 grids (rapid exploration)
3. Click → generate → click → generate
4. Follow aesthetic "threads" 5-6 levels deep
5. Use back button to try alternate paths
6. Discover unexpected transformations
```

## Files Added

```
src/diffusion_art/core/exploration.py      # Core exploration algorithms
src/diffusion_art/ui/photography_tab.py    # Streamlit UI
PHOTOGRAPHY_STUDIO.md                      # User guide
docs/photography_studio_example.md         # Walkthrough examples
docs/exploration_api.md                    # API reference
```

## Files Modified

```
app.py                                     # Added photography tab
src/diffusion_art/ui/__init__.py          # Export new tab
README.md                                  # Updated features
```

## Technical Details

### Exploration Algorithm

The core `generate_latent_grid()` function:
1. Takes a center latent point
2. Generates N² random directions in latent space
3. Scales by "exploration radius" and latent magnitude
4. Returns list of latent tensors ready for decoding

**Adaptive scaling**: Automatically adjusts step size based on the latent's own statistics, making the radius parameter intuitive across different images.

### Performance Profile

- **Latent generation**: Instant (<1ms for 25 variations)
- **Decoding bottleneck**: ~2 seconds per image on M1/M2 Mac
- **Total time**: Grid size × 2 seconds

Grid recommendations:
- **3×3 (9 images)**: Rapid exploration mode (~20s)
- **4×4 (16 images)**: Sweet spot (~35s) ⭐
- **5×5 (25 images)**: Thorough search (~55s)

## Use Cases

### 🎨 For Artists
- Generate collections of thematically related images
- Find unexpected interpretations of a concept
- Create "evolution series" documenting exploration paths

### 🔬 For Researchers
- Map latent space topology
- Document semantic neighborhoods
- Study model behavior in different regions

### 🎮 For Creative Technologists
- Interactive installations (click to explore)
- Generative art systems
- Latent space navigation interfaces

## What's Next

Potential future enhancements:
- [ ] PCA-guided exploration (semantically meaningful directions)
- [ ] Export favorite images gallery
- [ ] Side-by-side comparison mode
- [ ] Latent space "heat map" visualization
- [ ] Multi-image constellation mode
- [ ] Automated "interesting region" detection

## Try It Now

```bash
streamlit run app.py
```

Then select **📷 Photography Studio** from the tab selector.

---

**Philosophy**: This mode embraces the constraint that VAE decoding is slow. Instead of fighting it with optimizations, we make **curation** the creative act. You're not watching the computer generate art—you're hunting for it.
