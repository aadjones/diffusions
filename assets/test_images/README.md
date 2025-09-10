# Test Image Pairs for Latent Interpolation

This folder contains curated image pairs designed to showcase different types of latent space interpolation behavior.

## Organization Structure

Each test pair is stored as:
```
{category}_{description}_A.{ext}  # Start image
{category}_{description}_B.{ext}  # End image
```

## Recommended Test Pairs

### **Animals** - Species transformation
- `animals_cat-dog_A.jpg` / `animals_cat-dog_B.jpg`
- `animals_bird-fish_A.jpg` / `animals_bird-fish_B.jpg`

### **Portraits** - Human face morphing
- `portraits_young-old_A.jpg` / `portraits_young-old_B.jpg`
- `portraits_male-female_A.jpg` / `portraits_male-female_B.jpg`

### **Landscapes** - Environmental transitions
- `landscapes_mountain-beach_A.jpg` / `landscapes_mountain-beach_B.jpg`
- `landscapes_day-night_A.jpg` / `landscapes_day-night_B.jpg`

### **Architecture** - Structural morphing
- `architecture_modern-classical_A.jpg` / `architecture_modern-classical_B.jpg`
- `architecture_interior-exterior_A.jpg` / `architecture_interior-exterior_B.jpg`

### **Art Styles** - Artistic transformation
- `styles_photo-painting_A.jpg` / `styles_photo-painting_B.jpg`
- `styles_realistic-abstract_A.jpg` / `styles_realistic-abstract_B.jpg`

### **Extreme** - Test interpolation limits
- `extreme_organic-geometric_A.jpg` / `extreme_organic-geometric_B.jpg`
- `extreme_bright-dark_A.jpg` / `extreme_bright-dark_B.jpg`

## Usage Notes

- All images should be high quality (preferably 512×512 or larger)
- Avoid heavily watermarked or copyrighted images
- Choose images that highlight different interpolation behaviors
- SLERP vs LERP differences are most visible with very different subjects
