"""Photography Studio tab - explore latent space through contact sheets."""

from typing import List, Optional

import streamlit as st
import torch
from PIL import Image

from ..core.exploration import (
    generate_directional_grid,
    generate_latent_grid,
    refine_latent_region,
)
from ..models.vae import SD15VAE
from ..utils import presets


def _batch_decode_latents(
    vae_model: SD15VAE, latents: List[torch.Tensor], batch_size: int = 4
) -> List[Image.Image]:
    """Decode latents in batches for efficiency.

    Args:
        vae_model: VAE model for decoding
        latents: List of latent tensors to decode
        batch_size: Number to decode at once

    Returns:
        List of decoded images
    """
    images = []

    # Process in batches
    for i in range(0, len(latents), batch_size):
        batch = latents[i : i + batch_size]

        # Decode batch
        for latent in batch:
            img = vae_model.decode(latent)
            images.append(img)

    return images


def _display_contact_sheet(
    images: List[Image.Image],
    grid_size: int,
    caption: str,
    key_prefix: str = "",
    current_selection: Optional[int] = None,
) -> Optional[int]:
    """Display images in a grid and return selected index if clicked.

    Args:
        images: List of PIL images
        grid_size: Size of grid (grid_size x grid_size)
        caption: Caption for the contact sheet
        key_prefix: Unique prefix for button keys to avoid duplicates
        current_selection: Index of currently selected image (highlighted)

    Returns:
        Index of selected image, or None if none selected
    """
    st.subheader(caption)

    # Create grid layout
    selected_idx = None

    for row in range(grid_size):
        cols = st.columns(grid_size)
        for col_idx, col in enumerate(cols):
            img_idx = row * grid_size + col_idx

            if img_idx < len(images):
                with col:
                    # Highlight selected image
                    if img_idx == current_selection:
                        st.markdown("**🎯 SELECTED**")

                    # Display image
                    st.image(images[img_idx], width="stretch")

                    # Show selection state
                    if img_idx == current_selection:
                        button_label = "✅ Selected"
                        button_type = "primary"
                    else:
                        button_label = "📍 Select"
                        button_type = "secondary"

                    # Add button to select this image
                    if st.button(
                        button_label,
                        key=f"{key_prefix}_select_{img_idx}",
                        type=button_type,
                        help="Select this as center point for next exploration",
                    ):
                        selected_idx = img_idx

    return selected_idx


def render_photography_tab(vae_model: SD15VAE) -> None:  # noqa: C901
    """Render the photography studio exploration tab."""

    st.header("📷 Latent Space Photography Studio")
    st.write(
        "Explore latent space by generating grids of variations. "
        "Click any image to zoom in and explore that region in detail."
    )

    # Initialize session state
    if "photo_base_latent" not in st.session_state:
        st.session_state.photo_base_latent = None

    if "photo_current_latent" not in st.session_state:
        st.session_state.photo_current_latent = None

    if "photo_latent_history" not in st.session_state:
        st.session_state.photo_latent_history = []

    if "photo_current_seed" not in st.session_state:
        st.session_state.photo_current_seed = 42

    if "photo_exploration_depth" not in st.session_state:
        st.session_state.photo_exploration_depth = 0

    if "photo_selected_idx" not in st.session_state:
        st.session_state.photo_selected_idx = None

    if "photo_selected_latent" not in st.session_state:
        st.session_state.photo_selected_latent = None

    # === PHASE 1: IMAGE SELECTION ===
    st.subheader("📁 Starting Image")

    available_pairs = presets.get_available_pairs()
    base_img = None

    # Image source selection
    if available_pairs:
        col_preset, col_custom = st.columns([2, 1])
        with col_preset:
            preset_options = (
                ["Custom Upload"]
                + [f"{pair['name']} (A)" for pair in available_pairs]
                + [f"{pair['name']} (B)" for pair in available_pairs]
            )
            selected_option = st.selectbox(
                "Image Source",
                preset_options,
                index=1 if len(preset_options) > 1 else 0,
                help="Choose a preset or upload your own",
            )
        with col_custom:
            st.write("")
            use_custom = selected_option == "Custom Upload"
    else:
        use_custom = True

    # Handle preset vs custom
    if not use_custom and available_pairs:
        try:
            if " (A)" in selected_option:
                preset_name = selected_option.replace(" (A)", "")
                preset_pair = next(
                    p for p in available_pairs if p["name"] == preset_name
                )
                base_img, _ = presets.load_pair(preset_pair)
            elif " (B)" in selected_option:
                preset_name = selected_option.replace(" (B)", "")
                preset_pair = next(
                    p for p in available_pairs if p["name"] == preset_name
                )
                _, base_img = presets.load_pair(preset_pair)

            if base_img:
                st.image(base_img, caption="📷 Starting Image", width="stretch")
        except Exception as e:
            st.error(f"❌ Error loading preset: {str(e)}")
            use_custom = True

    if use_custom or base_img is None:
        uploaded = st.file_uploader(
            "Upload starting image",
            type=["png", "jpg", "jpeg"],
            key="photo_upload",
            help="Image to start exploring from",
        )

        if uploaded is None:
            st.info("👆 Upload an image to begin latent space photography")
            return

        base_img = Image.open(uploaded)
        st.image(base_img, caption="📷 Starting Image", width="stretch")

    # === PHASE 2: ENCODE BASE IMAGE ===
    st.subheader("🔄 Processing")

    # Only encode if this is a new image (check if base_latent is None or image changed)
    # This prevents overwriting photo_current_latent during exploration
    if st.session_state.photo_base_latent is None:
        with st.spinner("Encoding image to latent space..."):
            base_latent = vae_model.encode(base_img)
            st.session_state.photo_base_latent = base_latent
            st.session_state.photo_current_latent = base_latent
            st.success("✅ Image encoded successfully")
    else:
        st.success("✅ Image encoded successfully")

    # === PHASE 3: EXPLORATION CONTROLS ===
    st.subheader("🎛️ Exploration Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        grid_size = st.selectbox(
            "Grid Size",
            [3, 4, 5],
            index=1,
            help="3×3 = 9 images (fast), 4×4 = 16 images (balanced), 5×5 = 25 images (slow)",
        )

    with col2:
        exploration_mode = st.selectbox(
            "Exploration Mode",
            ["Chaos", "Hybrid", "Walk", "Directional", "Refined Zoom"],
            help=(
                "Chaos: Massive random noise (dramatic changes)\n"
                "Hybrid: Blend with random images (creates fusions)\n"
                "Walk: Random walk from center (accumulating drift)\n"
                "Directional: Organized radial pattern\n"
                "Refined Zoom: Subtle variations for fine-tuning"
            ),
        )

    with col3:
        exploration_radius = st.slider(
            "Exploration Radius",
            0.1,
            3.0,
            1.0,
            0.1,
            help="How far from center to explore (larger = more dramatic)",
        )

    # Seed control and navigation
    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("🎲 New Random Seed", key="new_random_seed"):
            st.session_state.photo_current_seed += 1
            st.rerun()

    with col5:
        if st.button(
            "🏠 Reset to Start", key="reset_to_start", help="Go back to original image"
        ):
            st.session_state.photo_current_latent = st.session_state.photo_base_latent
            st.session_state.photo_latent_history = []
            st.session_state.photo_exploration_depth = 0
            st.rerun()

    with col6:
        if len(st.session_state.photo_latent_history) > 0 and st.button(
            "⬅️ Go Back", key="go_back", help="Return to previous exploration"
        ):
            st.session_state.photo_current_latent = (
                st.session_state.photo_latent_history.pop()
            )
            st.session_state.photo_exploration_depth -= 1
            st.rerun()

    # Show exploration depth and current center point
    if st.session_state.photo_exploration_depth > 0:
        st.info(
            f"🗺️ Exploration depth: {st.session_state.photo_exploration_depth} "
            f"(History: {len(st.session_state.photo_latent_history)} steps)"
        )

        # Show preview of what we're exploring from
        st.write("**🎯 Currently exploring from:**")
        col_preview, col_info = st.columns([1, 2])
        with col_preview:
            # Decode and show current center point
            with st.spinner("Loading current position..."):
                center_preview = vae_model.decode(st.session_state.photo_current_latent)
                st.image(center_preview, caption="Center Point", width="stretch")
        with col_info:
            st.write(f"**Depth:** Level {st.session_state.photo_exploration_depth}")
            st.write(f"**Seed:** {st.session_state.photo_current_seed}")
            st.write("")
            st.write(
                "👇 Generate a new contact sheet to see variations around this image"
            )

    # === PHASE 4: GENERATE CONTACT SHEET ===
    st.subheader("🖼️ Contact Sheet")

    total_images = grid_size * grid_size
    estimated_time = total_images * 2  # ~2 seconds per decode

    st.info(
        f"📊 Generating {total_images} variations "
        f"(~{estimated_time // 60}m {estimated_time % 60}s)"
    )

    if st.button("🎬 Generate Contact Sheet", type="primary", key="generate_sheet"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Generate latent variations
            status_text.text("Generating latent variations...")
            current_latent = st.session_state.photo_current_latent

            if exploration_mode == "Chaos":
                latents = generate_latent_grid(
                    current_latent,
                    grid_size=grid_size,
                    exploration_radius=exploration_radius,
                    seed=st.session_state.photo_current_seed,
                    mode="chaos",
                )
            elif exploration_mode == "Hybrid":
                latents = generate_latent_grid(
                    current_latent,
                    grid_size=grid_size,
                    exploration_radius=exploration_radius,
                    seed=st.session_state.photo_current_seed,
                    mode="hybrid",
                )
            elif exploration_mode == "Walk":
                latents = generate_latent_grid(
                    current_latent,
                    grid_size=grid_size,
                    exploration_radius=exploration_radius,
                    seed=st.session_state.photo_current_seed,
                    mode="walk",
                )
            elif exploration_mode == "Directional":
                latents = generate_directional_grid(
                    current_latent,
                    grid_size=grid_size,
                    exploration_radius=exploration_radius,
                )
            elif exploration_mode == "Refined Zoom":
                latents = refine_latent_region(
                    current_latent,
                    grid_size=grid_size,
                    zoom_factor=exploration_radius * 0.3,
                    seed=st.session_state.photo_current_seed,
                )

            progress_bar.progress(0.2)

            # Decode latents one by one with progress
            status_text.text("Decoding latents to images...")
            images = []

            for i, latent in enumerate(latents):
                img = vae_model.decode(latent)
                images.append(img)

                # Update progress
                decode_progress = 0.2 + (0.8 * (i + 1) / len(latents))
                progress_bar.progress(decode_progress)
                status_text.text(f"Decoded {i + 1}/{len(latents)} images...")

            progress_bar.progress(1.0)
            status_text.text("✅ Contact sheet ready!")

            # Store in session state for interaction
            st.session_state.photo_current_images = images
            st.session_state.photo_current_latents = latents
            st.session_state.photo_current_grid_size = grid_size

            st.success("✅ Contact sheet generated! Scroll down to explore.")

        except Exception as e:
            st.error(f"❌ Error generating contact sheet: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

    # === PHASE 5: DISPLAY EXISTING CONTACT SHEET (if available) ===
    if "photo_current_images" in st.session_state:
        images = st.session_state.photo_current_images
        latents = st.session_state.photo_current_latents
        display_grid_size = st.session_state.get("photo_current_grid_size", grid_size)

        # Show clear instructions
        st.write("**👇 Click any image to select it as your new center point**")

        selected_idx = _display_contact_sheet(
            images,
            display_grid_size,
            f"📸 Contact Sheet (Seed: {st.session_state.photo_current_seed})",
            key_prefix="current",
            current_selection=st.session_state.photo_selected_idx,
        )

        # Handle selection
        if selected_idx is not None:
            # Store BOTH the index and the actual latent tensor
            st.session_state.photo_selected_idx = selected_idx
            st.session_state.photo_selected_latent = latents[selected_idx]
            st.rerun()

        # Show "Explore From Selected" button
        if st.session_state.photo_selected_idx is not None:
            st.write("")  # Spacing
            col_explore, col_cancel = st.columns([3, 1])

            with col_explore:
                if st.button(
                    f"🚀 Explore From Image #{st.session_state.photo_selected_idx + 1}",
                    type="primary",
                    key="explore_from_selected",
                    help="Generate new variations around the selected image",
                ):
                    # Save current position to history
                    st.session_state.photo_latent_history.append(
                        st.session_state.photo_current_latent
                    )

                    # Move to selected latent (use stored latent, not index lookup)
                    st.session_state.photo_current_latent = (
                        st.session_state.photo_selected_latent
                    )
                    st.session_state.photo_exploration_depth += 1

                    # Clear selection and current images to force new generation
                    st.session_state.photo_selected_idx = None
                    st.session_state.photo_selected_latent = None
                    del st.session_state.photo_current_images
                    del st.session_state.photo_current_latents

                    st.success(
                        "✅ Now exploring from selected image! "
                        "Click 'Generate Contact Sheet' above to see new variations."
                    )
                    st.rerun()

            with col_cancel:
                if st.button("❌ Cancel", key="cancel_selection"):
                    st.session_state.photo_selected_idx = None
                    st.session_state.photo_selected_latent = None
                    st.rerun()

    # === PHASE 6: EXPORT & TIPS ===
    with st.expander("💡 How to Use", expanded=False):
        st.write(
            """
        **Basic Workflow:**
        1. Click "Generate Contact Sheet" to create variations
        2. Click "📍 Select" under an interesting image
        3. Click "🚀 Explore From Image #X" to dive deeper
        4. Repeat! Use "⬅️ Go Back" if you hit a dead end

        **Exploration Tips:**
        - **Start broad, then zoom**: Use large radius (2-3) initially, then switch to "Refined Zoom" mode
        - **Random seed**: Change seed (🎲) to get different variations from the same point
        - **Grid size**: 3×3 is fastest for rapid exploration, 4×4 balanced, 5×5 thorough

        **Exploration Modes:**
        - **Chaos**: Massive random noise - wildly different images every time
        - **Hybrid**: Blend your image with random ones - creates surreal fusions
        - **Walk**: Random walk accumulating changes - progressive drift
        - **Directional**: Organized radial pattern (systematic coverage)
        - **Refined Zoom**: Subtle variations for fine-tuning (after finding something interesting)

        **What to Look For:**
        - Images that "surprise" you - unexpected interpretations
        - Coherent transformations that maintain recognizable features
        - Aesthetic threads worth following deeper

        **Performance:**
        - Each image takes ~2 seconds to decode
        - 3×3 grid = 9 images ≈ 20 seconds
        - 4×4 grid = 16 images ≈ 35 seconds
        - 5×5 grid = 25 images ≈ 55 seconds
        """
        )
