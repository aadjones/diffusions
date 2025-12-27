# Exploration Modes Explained

The Photography Studio now offers multiple ways to explore latent space. Each mode creates different kinds of variations:

## 🌪️ Chaos Mode
**What it does:** Adds massive random noise to your image

**Results:** Wildly different images each time. Dramatic transformations. Some will be unrecognizable, others surprisingly coherent.

**Best for:**
- Initial broad exploration
- Finding completely unexpected interpretations
- When you want maximum diversity

**Radius recommendations:**
- 0.5-1.0: Still dramatic but more coherent
- 1.5-2.0: Wild chaos, hit or miss
- 3.0: Complete destruction (for fun)

## 🎨 Hybrid Mode
**What it does:** Blends your image with completely random images from the latent distribution

**Results:** Creates surreal "fusion" images. Your subject merged with random content. Like multiple-exposure photography.

**Best for:**
- Creating dreamlike, surreal variations
- When you want your subject "infected" with other content
- Artistic remixes

**Radius recommendations:**
- 0.3: Subtle blending (70% your image, 30% random)
- 0.5: Equal blend (50/50 mix)
- 1.0: Mostly random (your image as faint influence)

## 🚶 Walk Mode
**What it does:** Takes a random walk through latent space, where each step builds on the previous one

**Results:** Progressive drift. Each image is a small step from the last, creating a "path" through latent space.

**Best for:**
- Following aesthetic threads
- Smooth progressions
- When you want variation but not chaos

**Radius recommendations:**
- 0.5: Slow, subtle drift
- 1.0: Medium pace evolution
- 2.0: Fast, dramatic changes between steps

**Note:** Walk mode creates a sequence - image #1 is close to center, image #16 is many steps away.

## 🎯 Directional Mode
**What it does:** Systematic radial exploration with organized pattern

**Results:** Variations arranged by direction from center

**Best for:**
- Systematic coverage of latent space
- Comparing different "axes" of variation
- Methodical exploration

## 🔬 Refined Zoom Mode
**What it does:** Tiny variations around your current point (the original "filter" behavior)

**Results:** Subtle tweaks. Good for fine-tuning after you've found something interesting with another mode.

**Best for:**
- Final polish
- Finding the "perfect" version of a good find
- Comparing minor differences

**Radius recommendations:**
- 0.1-0.3: Very subtle (original behavior)
- 0.5-1.0: More noticeable but still conservative

---

## Recommended Workflow

1. **Start with Chaos (radius 1.0-1.5)** → Find interesting directions
2. **Click the most surprising result** → Dive deeper
3. **Try Hybrid (radius 0.5)** → See what fusions emerge
4. **Switch to Walk (radius 1.0)** → Follow aesthetic threads
5. **Finish with Refined Zoom (radius 0.3)** → Perfect the final result

---

## Technical Details

- **Chaos**: `new_latent = center + (random_noise * 3.0 * magnitude * radius)`
- **Hybrid**: `new_latent = (1-radius) * center + radius * random_latent`
- **Walk**: Sequential random walk from center, accumulating steps
- **Directional**: Angular-organized random offsets
- **Refined Zoom**: Conservative random offsets (0.3× scaling)

---

## What Changed from Original

The original "Random Variations" mode was too conservative (using 0.3× scaling), creating variations that looked nearly identical. The new modes give you:

1. **More dramatic changes** (Chaos uses 10× more noise)
2. **Different exploration strategies** (Hybrid, Walk)
3. **Better control** over how variations are generated

Try all the modes! They each create completely different kinds of discoveries.
