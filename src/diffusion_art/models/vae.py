"""VAE encoding and decoding functionality for Stable Diffusion 1.5."""

from typing import List, Optional, cast

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image
from torchvision import transforms


class SD15VAE:
    """Stable Diffusion 1.5 VAE encoder/decoder."""

    SD15_SCALE = 0.18215

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize the VAE model.

        Args:
            device: Device to load model on. If None, auto-detects MPS or CPU.
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.vae: Optional[AutoencoderKL] = None
        self._transform = self._create_transform()

    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing transform."""
        return transforms.Compose(
            [
                transforms.Resize(
                    (512, 512), interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def load_model(self) -> None:
        """Load the VAE model if not already loaded."""
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="vae"
            )
            self.vae = self.vae.to(self.device)  # pyright: ignore
            self.vae = self.vae.eval()

    def encode(self, img: Image.Image) -> torch.Tensor:
        """Encode an image to latent space.

        Args:
            img: PIL Image to encode

        Returns:
            Latent tensor of shape (1, 4, 64, 64)
        """
        self.load_model()
        assert self.vae is not None, "VAE model not loaded"
        tensor = cast(torch.Tensor, self._transform(img.convert("RGB")))
        x = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.vae.encode(x)
            return encoded.latent_dist.sample() * self.SD15_SCALE  # pyright: ignore

    def decode(self, z: torch.Tensor) -> Image.Image:
        """Decode latent tensor to image.

        Args:
            z: Latent tensor of shape (1, 4, 64, 64)

        Returns:
            PIL Image
        """
        self.load_model()
        assert self.vae is not None, "VAE model not loaded"
        with torch.no_grad():
            decoded = self.vae.decode(z / self.SD15_SCALE)  # pyright: ignore
            x = decoded.sample  # pyright: ignore
        x = (x.clamp(-1, 1) + 1) / 2
        arr = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def decode_batch(
        self, latent_batch: List[torch.Tensor], batch_size: int = 8
    ) -> List[Image.Image]:
        """Decode multiple latent tensors efficiently in batches.

        Args:
            latent_batch: List of latent tensors, each shape (1, 4, 64, 64)
            batch_size: Number of latents to process simultaneously

        Returns:
            List of PIL Images
        """
        print(
            f"🔄 Starting batch decode: {len(latent_batch)} frames, batch_size={batch_size}"
        )
        self.load_model()
        assert self.vae is not None, "VAE model not loaded"
        print(f"✅ VAE model loaded on {self.device}")

        images = []
        total_batches = (len(latent_batch) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(latent_batch), batch_size)):
                print(f"📦 Processing batch {batch_idx + 1}/{total_batches}")

                # Create batch by stacking individual latents
                batch_end = min(i + batch_size, len(latent_batch))
                current_batch = latent_batch[i:batch_end]
                print(f"   Batch size: {len(current_batch)} tensors")

                # Stack (N, 1, 4, 64, 64) -> (N, 4, 64, 64)
                print("   Stacking tensors...")
                batch_tensor = torch.cat(current_batch, dim=0)
                print(f"   Batch tensor shape: {batch_tensor.shape}")

                # Decode entire batch at once
                print("   Running VAE decode...")
                scaled_tensor = cast(torch.FloatTensor, batch_tensor / self.SD15_SCALE)
                decoded = self.vae.decode(scaled_tensor)
                x = decoded.sample  # pyright: ignore
                print("   VAE decode complete")
                x = (x.clamp(-1, 1) + 1) / 2

                # Convert each image in batch
                print("   Converting to PIL images...")
                for j in range(x.shape[0]):
                    arr = (x[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    images.append(Image.fromarray(arr))

                print(f"   ✅ Batch {batch_idx + 1} complete")

        print(f"🎉 All batches complete! Generated {len(images)} images")
        return images
