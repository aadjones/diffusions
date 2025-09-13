Quick Start Guide
=================

Basic Usage
-----------

The easiest way to use Diffusion Art is through the Streamlit web interface:

.. code-block:: bash

   streamlit run app.py

This launches an interactive web interface where you can:

1. Upload multiple images (512×512 recommended)
2. Encode images to latent space
3. Interpolate between latent representations
4. Decode back to images
5. Download results

Programmatic Usage
------------------

You can also use Diffusion Art programmatically:

.. code-block:: python

   from diffusion_art.models.vae import SD15VAE
   from diffusion_art.core.interpolation import slerp
   from PIL import Image
   import torch

   # Load the VAE model
   vae = SD15VAE()

   # Load and encode images
   img1 = Image.open("image1.jpg")
   img2 = Image.open("image2.jpg")

   latent1 = vae.encode(img1)
   latent2 = vae.encode(img2)

   # Interpolate in latent space
   interpolated = slerp(latent1, latent2, 0.5)  # 50% blend

   # Decode back to image
   result = vae.decode(interpolated)
   result.save("interpolated.jpg")

Key Concepts
------------

**Latent Space**
  4-channel tensors of size 64×64 representing 512×512 images with 8× compression.

**SLERP vs LERP**
  - SLERP (Spherical Linear Interpolation): Preserves magnitude, better for most cases
  - LERP (Linear Interpolation): Direct linear blending, faster computation

**Device Support**
  Auto-detects MPS (Apple Silicon) or falls back to CPU for optimal performance.

Common Workflows
----------------

**Image Morphing**
  Create smooth transitions between two images using SLERP interpolation.

**Style Blending**
  Combine artistic styles by interpolating between encoded style references.

**Latent Space Exploration**
  Navigate the continuous latent space to discover variations and combinations.
