import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

from diffusion_art.models.vae import SD15VAE
from diffusion_art.ui import render_breathing_tab, render_interpolation_tab


@st.cache_resource
def load_vae_model() -> SD15VAE:
    vae = SD15VAE()
    vae.load_model()
    return vae


vae_model = load_vae_model()

st.title("🎨 Diffusion Latent Space Explorer")
st.caption(
    "Interactive interpolation between images in Stable Diffusion's latent space"
)

# Create tabs for different experiments
tab1, tab2 = st.tabs(["🔄 Interpolation", "🌊 Latent Breathing"])

with tab1:
    render_interpolation_tab(vae_model)

with tab2:
    render_breathing_tab(vae_model)
