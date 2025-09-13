Examples
========

Basic Image Interpolation
--------------------------

This example shows how to interpolate between two images:

.. code-block:: python

   from diffusion_art.models.vae import SD15VAE
   from diffusion_art.core.interpolation import slerp
   from PIL import Image

   # Initialize VAE
   vae = SD15VAE()

   # Load images
   img1 = Image.open("portrait1.jpg").convert("RGB").resize((512, 512))
   img2 = Image.open("portrait2.jpg").convert("RGB").resize((512, 512))

   # Encode to latent space
   latent1 = vae.encode(img1)
   latent2 = vae.encode(img2)

   # Create interpolation sequence
   steps = 10
   for i in range(steps + 1):
       t = i / steps
       interpolated = slerp(latent1, latent2, t)
       result = vae.decode(interpolated)
       result.save(f"frame_{i:03d}.jpg")

Multi-way Blending
------------------

Blend between multiple images simultaneously:

.. code-block:: python

   from diffusion_art.models.vae import SD15VAE
   from diffusion_art.core.interpolation import multi_slerp
   from PIL import Image
   import torch

   vae = SD15VAE()

   # Load multiple images
   images = [
       Image.open(f"style_{i}.jpg").convert("RGB").resize((512, 512))
       for i in range(4)
   ]

   # Encode all images
   latents = [vae.encode(img) for img in images]

   # Define weights (must sum to 1.0)
   weights = torch.tensor([0.4, 0.3, 0.2, 0.1])

   # Blend
   blended = multi_slerp(latents, weights)
   result = vae.decode(blended)
   result.save("blended_result.jpg")

Batch Processing
----------------

Process multiple interpolations efficiently:

.. code-block:: python

   from diffusion_art.models.vae import SD15VAE
   from diffusion_art.core.interpolation import slerp
   import torch
   from PIL import Image

   vae = SD15VAE()

   # Load source and target images
   sources = [Image.open(f"source_{i}.jpg") for i in range(5)]
   targets = [Image.open(f"target_{i}.jpg") for i in range(5)]

   # Encode in batches for efficiency
   source_latents = torch.stack([vae.encode(img) for img in sources])
   target_latents = torch.stack([vae.encode(img) for img in targets])

   # Interpolate
   interpolated = slerp(source_latents, target_latents, 0.5)

   # Decode results
   for i, latent in enumerate(interpolated):
       result = vae.decode(latent)
       result.save(f"interpolated_{i}.jpg")

Animation Creation
------------------

Create smooth animations with consistent frame timing:

.. code-block:: python

   from diffusion_art.models.vae import SD15VAE
   from diffusion_art.core.interpolation import slerp
   from PIL import Image
   import numpy as np

   def create_loop_animation(images, frames_per_transition=30):
       vae = SD15VAE()

       # Encode all keyframe images
       latents = [vae.encode(img) for img in images]

       all_frames = []

       # Interpolate between each pair of keyframes
       for i in range(len(latents)):
           start = latents[i]
           end = latents[(i + 1) % len(latents)]  # Loop back to first

           for frame in range(frames_per_transition):
               t = frame / frames_per_transition
               interpolated = slerp(start, end, t)
               decoded = vae.decode(interpolated)
               all_frames.append(decoded)

       return all_frames

   # Usage
   keyframes = [Image.open(f"key_{i}.jpg") for i in range(4)]
   animation_frames = create_loop_animation(keyframes)

   for i, frame in enumerate(animation_frames):
       frame.save(f"animation_frame_{i:04d}.jpg")
