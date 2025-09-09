import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from PIL import Image

from diffusion_art.core.interpolation import slerp
from diffusion_art.models.vae import SD15VAE


@st.cache_resource
def load_vae_model() -> SD15VAE:
    vae = SD15VAE()
    vae.load_model()
    return vae


vae_model = load_vae_model()

st.title("Latent Interpolation — SD 1.5 VAE")
col1, col2 = st.columns(2)
up1 = col1.file_uploader("Image A", type=["png", "jpg", "jpeg"])
up2 = col2.file_uploader("Image B", type=["png", "jpg", "jpeg"])

if up1 and up2:
    imgA, imgB = Image.open(up1), Image.open(up2)
    zA, zB = vae_model.encode(imgA), vae_model.encode(imgB)
    t = st.slider("Interpolation t", 0.0, 1.0, 0.0, 0.01)
    out = vae_model.decode(slerp(zA, zB, t))
    col3, col4 = st.columns(2)
    col3.image(imgA, caption="Image A", width="stretch")
    col4.image(imgB, caption="Image B", width="stretch")
    st.image(out, caption=f"Interpolated (t={t:.2f})", width="stretch")
else:
    st.info("Upload two images to begin.")
