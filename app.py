import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

from diffusion_art.config import features
from diffusion_art.core.interpolation import create_interpolation_path, lerp, slerp
from diffusion_art.models.vae import SD15VAE


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

# === PHASE 1: IMAGE UPLOAD ===
st.header("📁 Source Images")
st.write("Upload two images to interpolate between. Both will be resized to 512×512.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🅰️ Start Image")
    up1 = st.file_uploader(
        "Choose starting image", type=["png", "jpg", "jpeg"], key="img_a"
    )

with col2:
    st.subheader("🅱️ End Image")
    up2 = st.file_uploader(
        "Choose ending image", type=["png", "jpg", "jpeg"], key="img_b"
    )

if not up1 or not up2:
    st.info("👆 Upload both images to begin latent space exploration")
    st.stop()

# === PHASE 2: PROCESSING & PREVIEW ===
st.header("🔄 Processing")

with st.spinner("Encoding images to latent space..."):
    try:
        imgA, imgB = Image.open(up1), Image.open(up2)
        zA, zB = vae_model.encode(imgA), vae_model.encode(imgB)
        st.success("✅ Images encoded successfully")
    except Exception as e:
        st.error(f"❌ Error processing images: {str(e)}")
        st.stop()

# Show source images
col3, col4 = st.columns(2)
col3.image(imgA, caption="🅰️ Start Image", width="stretch")
col4.image(imgB, caption="🅱️ End Image", width="stretch")

# === PHASE 3: CORE INTERPOLATION ===
st.header("🎛️ Interpolation Controls")

col_method, col_t = st.columns([1, 2])
with col_method:
    method = st.selectbox(
        "Method",
        ["SLERP", "LERP"],
        help="SLERP preserves magnitude, LERP is direct linear blend",
    )

with col_t:
    t = st.slider(
        "Interpolation Factor",
        0.0,
        1.0,
        0.5,
        0.01,
        help="0.0 = Start image, 1.0 = End image",
    )

# Generate interpolation
with st.spinner("Generating interpolation..."):
    interpolate_fn = slerp if method == "SLERP" else lerp
    z_interp = interpolate_fn(zA, zB, t)
    out = vae_model.decode(z_interp)

# Show result prominently
st.subheader(f"🎯 {method} Result (t={t:.2f})")
st.image(out, width="stretch")

# === PHASE 4: ADVANCED FEATURES ===
st.header("🧪 Advanced Tools")

# Quick comparison
if features.is_enabled("comparison_mode"):
    with st.expander("🔄 Method Comparison", expanded=False):
        st.write("Compare SLERP vs LERP side-by-side at the same t-value")

        slerp_out = vae_model.decode(slerp(zA, zB, t))
        lerp_out = vae_model.decode(lerp(zA, zB, t))

        col_comp1, col_comp2 = st.columns(2)
        col_comp1.image(slerp_out, caption=f"SLERP (t={t:.2f})", width="stretch")
        col_comp2.image(lerp_out, caption=f"LERP (t={t:.2f})", width="stretch")

        # Pixel difference visualization
        diff_array = np.abs(np.array(slerp_out) - np.array(lerp_out))
        diff_img = Image.fromarray(diff_array.astype(np.uint8))
        st.image(diff_img, caption="Difference Map (SLERP - LERP)", width="stretch")

# Animation export
with st.expander("🎬 Animation Export", expanded=False):
    st.write("Generate smooth interpolation animations as GIF files")

    col_anim1, col_anim2, col_anim3 = st.columns(3)
    with col_anim1:
        steps = st.slider("Frames", 5, 50, 20, help="Number of animation frames")
    with col_anim2:
        duration = st.slider("Frame Duration (ms)", 50, 500, 100)
    with col_anim3:
        loop_back = st.checkbox(
            "Loop Back", value=True, help="A→B→A for seamless loops"
        )

    if st.button("🎬 Generate Animation", type="primary"):
        with st.spinner("Generating animation frames..."):
            progress_bar = st.progress(0)

            # Create interpolation path
            if loop_back:
                path_forward = create_interpolation_path(
                    zA, zB, steps // 2 + 1, method.lower()
                )
                path_backward = create_interpolation_path(
                    zB, zA, steps // 2 + 1, method.lower()
                )[1:]
                latent_path = path_forward + path_backward
            else:
                latent_path = create_interpolation_path(zA, zB, steps, method.lower())

            # Generate frames
            frames = []
            for i, latent in enumerate(latent_path):
                frame = vae_model.decode(latent)
                frames.append(frame)
                progress_bar.progress((i + 1) / len(latent_path))

            # Create GIF
            gif_buffer = io.BytesIO()
            frames[0].save(
                gif_buffer,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )
            gif_buffer.seek(0)

            st.success("✅ Animation generated!")
            st.image(
                gif_buffer.getvalue(),
                caption=f"{method} Animation ({len(frames)} frames)",
            )

            # Download button
            st.download_button(
                label="📥 Download GIF",
                data=gif_buffer.getvalue(),
                file_name=f"latent_{method.lower()}_{steps}f_{duration}ms.gif",
                mime="image/gif",
                type="primary",
            )

# Debug panel for power users
if features.is_enabled("debug_panel"):
    with st.expander("🔍 Latent Space Metrics", expanded=False):
        col_debug1, col_debug2, col_debug3 = st.columns(3)

        with col_debug1:
            st.metric("Start Magnitude", f"{zA.norm().item():.3f}")
            st.metric("End Magnitude", f"{zB.norm().item():.3f}")
            st.metric("Result Magnitude", f"{z_interp.norm().item():.3f}")

        with col_debug2:
            cos_sim = torch.cosine_similarity(zA.flatten(), zB.flatten(), dim=0)
            st.metric("Cosine Similarity", f"{cos_sim.item():.3f}")
            st.metric("Latent Shape", f"{zA.shape}")

        with col_debug3:
            if method == "SLERP":
                linear_interp = lerp(zA, zB, t)
                interp_diff = (z_interp - linear_interp).norm().item()
                st.metric("SLERP vs LERP Δ", f"{interp_diff:.6f}")

            dist_A = (z_interp - zA).norm().item()
            dist_B = (z_interp - zB).norm().item()
            st.metric("Distance to Start", f"{dist_A:.3f}")
            st.metric("Distance to End", f"{dist_B:.3f}")
