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

# Initialize session state for tab selection
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "interpolation"

# Create tab selection with pills
tab_options = {"interpolation": "🔄 Interpolation", "breathing": "🌊 Latent Breathing"}

selected_tab = st.pills(
    "Choose experiment:",
    options=list(tab_options.keys()),
    format_func=lambda x: tab_options[x],
    default=st.session_state.active_tab,
    key="tab_selector",
)

# Update session state
st.session_state.active_tab = selected_tab

# Add some spacing
st.write("")

# Render the appropriate tab content
if selected_tab == "interpolation":
    render_interpolation_tab(vae_model)
elif selected_tab == "breathing":
    render_breathing_tab(vae_model)
