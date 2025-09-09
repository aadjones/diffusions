# Diffusion Latent Space Art Experiments — Setup & Key Ideas

## Conceptual Summary

We’ve been circling around **using diffusion models as artistic latent playgrounds**. The important takeaways:

- **Latent Space as Medium:** Stable Diffusion 1.5 (with its fixed VAE) defines a compressed 64×64×4 latent space. All our experiments happen here, not in pixel space.
- **Noise and Sculpting:** Training adds Gaussian noise to latents and teaches a UNet to subtract it, like a sculptor chiseling rubble away. Generation runs this process backward: start from noise, sculpt an image guided by text embeddings.
- **UNet as Sculptor:** The U-shaped architecture captures both global (composition) and local (texture) structure, refining over ~20–50 steps.
- **Intermediate Latents:** Decoding at different steps shows snapshots of the sculptor’s process: blocky bulk early, structure mid, details late.
- **Latent Walks:** Interpolating (LERP/SLERP) between encoded latents yields morphs — straight lines, loops, or curved paths through latent space.
- **Meta-latent Maps:** By encoding many images and mapping them with PCA/UMAP, we can visualize clusters and chart aesthetic journeys (geodesics, loops, phase boundaries).
- **Fisher Curvature:** Latent space isn’t flat — the Fisher information metric reveals smooth vs turbulent regions. High curvature zones = unpredictable interpolations (the “weather map” metaphor).

Artistic angle: treat latent navigation like **musical improvisation** — pick dimensions, fix some, vary others, trace paths (straight, looping, wandering). The artwork emerges not from prompts but from **geometry and constraint.**

---

## Practical Path Forward

Two useful modes for exploration on local hardware (M1 MacBook Pro, 2021):

1. **Notebook (Jupyter/VS Code):**
   - Best for research and quick iteration.
   - Encode → SLERP → Decode → Preview.
   - Sliders via ipywidgets let you scrub interpolation live.

2. **Streamlit App:**
   - Best for a clean, artist-friendly UI.
   - File upload for two images + slider to navigate interpolation.
   - Can expand later to atlas views, loops, audio-driven controls.

---

## Setting Up the Streamlit App on M1 Mac

### 1. Environment Setup

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision
pip install diffusers transformers accelerate safetensors pillow umap-learn
pip install streamlit
```

MPS/Metal is supported natively on macOS. Optional fallback flag:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 2. Streamlit App Code

Create `app.py`:

```python
import streamlit as st
import torch, numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

device = "mps" if torch.backends.mps.is_available() else "cpu"

@st.cache_resource
def load_vae():
    return AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    ).to(device).eval()

vae = load_vae()
SD15_SCALE = 0.18215

pre = transforms.Compose([
    transforms.Resize((512,512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

def encode(img: Image.Image):
    x = pre(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return vae.encode(x).latent_dist.sample() * SD15_SCALE

def decode(z: torch.Tensor):
    with torch.no_grad():
        x = vae.decode(z / SD15_SCALE).sample
    x = (x.clamp(-1,1)+1)/2
    arr = (x[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return Image.fromarray(arr)

def slerp(z0, z1, t: float):
    a, b = z0.flatten(1), z1.flatten(1)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    dot = (a*b).sum(dim=1, keepdim=True).clamp(-0.999999, 0.999999)
    theta = torch.acos(dot)
    sin_t = torch.sin(theta)
    w0 = torch.sin((1-t)*theta)/sin_t
    w1 = torch.sin(t*theta)/sin_t
    out = (w0*a + w1*b).view_as(z0)
    mag = z0.flatten(1).norm(dim=1, keepdim=True)
    out = out.flatten(1)
    out = out / (out.norm(dim=1, keepdim=True)+1e-8) * mag
    return out.view_as(z0)

st.title("Latent Interpolation — SD 1.5 VAE")
col1, col2 = st.columns(2)
up1 = col1.file_uploader("Image A", type=["png","jpg","jpeg"])
up2 = col2.file_uploader("Image B", type=["png","jpg","jpeg"])

if up1 and up2:
    imgA, imgB = Image.open(up1), Image.open(up2)
    zA, zB = encode(imgA), encode(imgB)
    t = st.slider("Interpolation t", 0.0, 1.0, 0.0, 0.01)
    out = decode(slerp(zA, zB, t))
    col3, col4 = st.columns(2)
    col3.image(imgA, caption="Image A", use_column_width=True)
    col4.image(imgB, caption="Image B", use_column_width=True)
    st.image(out, caption=f"Interpolated (t={t:.2f})", use_column_width=True)
else:
    st.info("Upload two images to begin.")
```

### 3. Run the App

```bash
streamlit run app.py
```

This opens a local web app where you can upload two images, scrub the latent interpolation slider, and preview decoded outputs.

---

## Next Steps

- Build **meta-latent atlases** with PCA/UMAP.
- Add **closed-loop interpolations** or multi-anchor paths.
- Map **MIDI input → latent coordinates** for cross-modal improvisation.
- Experiment with **Fisher curvature** maps to guide smooth vs. turbulent regions of latent exploration.
