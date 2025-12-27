# 📷 Photography Studio Guide

The Photography Studio is a new tab in Diffusion Art that lets you **explore latent space through contact sheets** - like a photographer reviewing film negatives.

## Core Concept

Instead of watching smooth animations, you **hunt for interesting moments** by:
1. Generate a grid of variations around a point in latent space
2. Pick the most interesting image
3. Generate variations around *that* image
4. Repeat, going deeper into interesting regions

This is like following a trail of breadcrumbs through the latent space landscape.

## How to Use

### 1. Start with an Image
- Upload any image or select from presets
- This becomes your "origin point" in latent space

### 2. Generate Contact Sheet
Click "Generate Contact Sheet" to create a grid (3×3, 4×4, or 5×5) of variations:

**Grid sizes:**
- 3×3 = 9 images (~20 seconds)
- 4×4 = 16 images (~35 seconds) **← recommended**
- 5×5 = 25 images (~55 seconds)

### 3. Explore Deeper
- Click "🔍 Explore" under any interesting image
- That image becomes your new center point
- Generate another contact sheet to explore that region
- Use "⬅️ Go Back" to backtrack if you hit a dead end

### 4. Exploration Modes

**Random Variations** (default)
- Chaotic exploration in all directions
- Good for initial broad search

**Directional**
- Organized radial pattern
- Good for systematic coverage

**Refined Zoom**
- Subtle variations for fine-tuning
- Use after finding a promising region

### 5. Controls

**Exploration Radius:**
- **0.1-0.5**: Subtle variations, stay close to center
- **1.0**: Balanced exploration (default)
- **2.0-3.0**: Wild variations, explore far from center

**Random Seed:**
- Same seed = same variations
- Click "🎲 New Random Seed" for different variations from same point

**Navigation:**
- "🏠 Reset to Start": Jump back to original image
- "⬅️ Go Back": Return to previous exploration step

## Exploration Strategies

### Strategy 1: Broad → Narrow
1. Start with **large radius (2.0)** and **Random Variations**
2. Find generally interesting direction
3. Switch to **Refined Zoom** with **small radius (0.5)**
4. Fine-tune the aesthetic

### Strategy 2: Systematic Survey
1. Use **Directional** mode with **4×4 grid**
2. Explore each "quadrant" of latent space
3. Map out what different regions look like

### Strategy 3: Following Threads
1. Generate 3×3 grid for **fast exploration**
2. Click most interesting → generate another 3×3
3. Keep clicking → following aesthetic "trails"
4. Use back button to try different branches

## What Makes a Good Exploration?

Look for images that:
- **Surprise you** - unexpected interpretations
- **Maintain coherence** - still recognizable but transformed
- **Suggest direction** - hint at interesting nearby regions
- **Have aesthetic unity** - consistent style/mood

## Performance Tips

- **3×3 grid is fastest** for rapid exploration (20 sec)
- **4×4 is sweet spot** for thoroughness vs speed (35 sec)
- **5×5 is slow** but comprehensive (55 sec)
- Each decode takes ~2 seconds on Apple Silicon

## Advanced: Understanding Exploration Depth

The "exploration depth" counter shows how many clicks deep you are:
- Depth 0 = original image
- Depth 1 = first click
- Depth 5 = five levels deep into latent space

Deeper ≠ better! Sometimes the most interesting images are just 1-2 clicks away.

## Example Workflow

```
1. Upload portrait photo
2. Generate 4×4 Random grid (radius 1.5)
3. Click image that looks dreamlike
4. Generate 3×3 Refined grid (radius 0.3)
5. Find perfect subtle variation
6. Export that image
```

## Comparison to Other Modes

| Mode | Speed | Output | Best For |
|------|-------|--------|----------|
| **Interpolation** | Fast | Smooth animation | Morphing between two images |
| **Breathing** | Slow | Video loops | Animated breathing effects |
| **Photography** | Medium | Still images | Finding unique moments |

## Future Enhancements

Possible additions (not yet implemented):
- Export favorites gallery
- Side-by-side comparison mode
- PCA-guided exploration
- Latent space "heat map" visualization
