"""Latent breathing tab functionality."""

import io
import time
from typing import Optional, Tuple

import streamlit as st
import torch
from PIL import Image

from ..core.noise import breathing_animation, structured_noise
from ..models.vae import SD15VAE
from ..utils import presets


def _handle_image_selection() -> Optional[Image.Image]:
    """Handle image selection phase and return selected image."""
    st.subheader("📷 Base Image")

    # Load available presets
    available_pairs = presets.get_available_pairs()

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
                "Base Image Source",
                preset_options,
                index=1 if len(preset_options) > 1 else 0,
                help="Choose a preset image or upload your own",
            )
        with col_custom:
            st.write("")  # Spacer
            use_custom = selected_option == "Custom Upload"
    else:
        use_custom = True
        st.write("Upload an image to use as the base for breathing effects.")

    # Handle preset loading vs custom upload
    base_img = None
    if not use_custom and available_pairs:
        try:
            # Parse preset selection
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
                st.image(base_img, caption="🫁 Base Image (Preset)", width="stretch")
                st.success(f"✅ Loaded preset: {selected_option}")

        except Exception as e:
            st.error(f"❌ Error loading preset: {str(e)}")
            use_custom = True

    if use_custom or base_img is None:
        uploaded_file = st.file_uploader(
            "Choose base image",
            type=["png", "jpg", "jpeg"],
            key="breathing_img",
            help="This image will be the center of the breathing effect",
        )

        if uploaded_file is None:
            st.info("👆 Upload or select an image to begin breathing experiment")
            return None

        base_img = Image.open(uploaded_file)
        st.image(base_img, caption="🫁 Base Image", width="stretch")

    return base_img


def _handle_breathing_controls() -> Tuple[float, str]:
    """Handle breathing controls UI and return control values."""
    st.subheader("🎛️ Breathing Controls")

    col1, col2 = st.columns(2)

    with col1:
        noise_strength = st.slider(
            "Noise Strength",
            0.0,
            5.0,
            1.0,
            0.1,
            help="How much the image 'breathes' (0.0 = no movement, 5.0 = wild)",
        )

    with col2:
        animation_type = st.selectbox(
            "Animation Type",
            [
                "distance_threshold",
                "standard",
                "momentum",
                "attracted_home",
                "deep_note",
                "sine",
                "heartbeat",
                "pulse",
            ],
            help="Type of latent space animation",
        )

    return noise_strength, animation_type


def _handle_preview_controls(
    animation_type: str,
) -> Tuple[Optional[int], int, Optional[float]]:
    """Handle preview controls UI and return preview values."""
    if animation_type in [
        "distance_threshold",
        "standard",
        "momentum",
        "attracted_home",
        "deep_note",
    ]:
        # Random walk controls
        col4, col5 = st.columns(2)
        with col4:
            preview_step = st.slider(
                "Preview Step", 0, 30, 15, 1, help="Which step of the walk to preview"
            )
        with col5:
            preview_seed = st.number_input(
                "Walk Seed",
                value=42,
                min_value=0,
                max_value=9999,
                help="Change seed for different walk patterns",
            )
        return preview_step, preview_seed, None
    else:
        # Breathing pattern controls
        col4, col5 = st.columns(2)
        with col4:
            manual_t = st.slider(
                "Manual Position",
                0.0,
                1.0,
                0.0,
                0.01,
                help="Manually control breathing position (0 = inhale, 1 = exhale)",
            )
        with col5:
            preview_seed = st.number_input(
                "Breathing Seed",
                value=42,
                min_value=0,
                max_value=9999,
                help="Change seed for different breathing patterns",
            )
        return None, preview_seed, manual_t


def _generate_and_display_preview(
    vae_model: SD15VAE,
    base_latent: torch.Tensor,
    animation_type: str,
    noise_strength: float,
    preview_step: Optional[int],
    preview_seed: int,
    manual_t: Optional[float],
) -> None:
    """Generate and display preview frame."""
    with st.spinner("Generating preview..."):
        try:
            if animation_type in [
                "distance_threshold",
                "standard",
                "momentum",
                "attracted_home",
                "deep_note",
            ]:
                # Random walk preview
                from ..core.random_walk import (
                    deep_note_walk,
                    distance_threshold_walk,
                    latent_random_walk,
                    momentum_random_walk,
                )

                # Generate walk path (cached in session state)
                walk_key = f"random_walk_{id(base_latent)}_{animation_type}_{noise_strength}_{preview_seed}"
                if walk_key not in st.session_state:
                    with st.spinner("Generating random walk path..."):
                        if animation_type == "distance_threshold":
                            walk_path, turn_around_step = distance_threshold_walk(
                                base_latent,
                                steps=31,
                                step_size=noise_strength,
                                explore_fraction=0.62,
                                seed=preview_seed,
                            )
                            # Store turn around info for display
                            st.session_state[f"{walk_key}_turnaround"] = (
                                turn_around_step
                            )
                        elif animation_type == "standard":
                            walk_path = latent_random_walk(
                                base_latent,
                                steps=31,
                                step_size=noise_strength,
                                seed=preview_seed,
                            )
                        elif animation_type == "momentum":
                            walk_path = momentum_random_walk(
                                base_latent,
                                steps=31,
                                step_size=noise_strength,
                                momentum=0.8,
                                seed=preview_seed,
                            )
                        elif animation_type == "attracted_home":
                            walk_path = latent_random_walk(
                                base_latent,
                                steps=31,
                                step_size=noise_strength,
                                return_home=True,
                                seed=preview_seed,
                            )
                        elif animation_type == "deep_note":
                            walk_path = deep_note_walk(
                                base_latent,
                                steps=31,
                                step_size=noise_strength,
                                seed=preview_seed,
                            )

                        st.session_state[walk_key] = walk_path

                # Get the specific step for preview
                walk_path = st.session_state[walk_key]
                step_index = preview_step if preview_step is not None else 0
                preview_latent = walk_path[min(step_index, len(walk_path) - 1)]

            else:
                # Breathing pattern preview (sine, heartbeat, pulse)
                from ..core.noise import add_gaussian_noise, get_pattern_function

                pattern_func = get_pattern_function(
                    animation_type
                )  # animation_type is now the pattern
                t_value = manual_t if manual_t is not None else 0.0
                current_strength = noise_strength * pattern_func(t_value)
                preview_latent = add_gaussian_noise(
                    base_latent, current_strength, seed=preview_seed
                )

            preview_img = vae_model.decode(preview_latent)

            if animation_type in [
                "distance_threshold",
                "standard",
                "momentum",
                "attracted_home",
                "deep_note",
            ]:
                caption = (
                    f"🚶 Random Walk Preview (Step {preview_step}, {animation_type})"
                )

                # Add turn around info for distance threshold walks
                if animation_type == "distance_threshold":
                    turn_around_key = f"{walk_key}_turnaround"
                    if turn_around_key in st.session_state:
                        turn_step = st.session_state[turn_around_key]
                        if preview_step <= turn_step:
                            caption += " [Exploring]"
                        else:
                            caption += " [Returning home via SLERP]"
                        st.write(f"🔄 Turned around at step {turn_step}")

                st.image(preview_img, caption=caption, width="stretch")
            else:
                t_display = manual_t if manual_t is not None else 0.0
                st.image(
                    preview_img,
                    caption=f"🌊 {animation_type.title()} Breathing Preview (t={t_display:.2f}, strength: {current_strength:.3f})",
                    width="stretch",
                )

        except Exception as e:
            st.error(f"❌ Error generating preview: {str(e)}")
            return


def _handle_animation_generation(
    vae_model: SD15VAE,
    base_latent: torch.Tensor,
    animation_type: str,
    noise_strength: float,
) -> None:
    """Handle animation generation UI and processing."""
    st.subheader("🎬 Generate Animation")

    # Time-based controls only - who thinks in frame duration milliseconds?!
    col6a, col6b, col8 = st.columns(3)
    with col6a:
        duration_seconds = st.slider(
            "Duration (seconds)", 1.0, 30.0, 7.0, 0.5, help="Total animation duration"
        )
    with col6b:
        fps = st.slider("Frame Rate", 10, 60, 24, 1, help="Frames per second")
    with col8:
        anim_seed = st.number_input(
            "Animation Seed", value=42, min_value=0, max_value=9999
        )

    # Calculate frames and frame duration from sensible inputs
    anim_frames = int(duration_seconds * fps)
    frame_duration = round(1000 / fps)  # Round instead of truncate for accurate timing
    st.info(f"📊 {anim_frames} frames at {fps} fps ({frame_duration}ms per frame)")
    print(
        f"🔢 UI Calculation: {duration_seconds}s × {fps}fps = {anim_frames} frames, {frame_duration}ms duration"
    )

    if st.button("🎬 Generate Animation", type="primary", key="generate_animation_btn"):
        with st.spinner("Generating animation..."):
            try:
                progress_bar = st.progress(0)

                # Generate animation frames
                if animation_type in [
                    "distance_threshold",
                    "standard",
                    "momentum",
                    "attracted_home",
                    "deep_note",
                ]:
                    # Use actual random walk path for animation
                    from ..core.random_walk import (
                        deep_note_walk,
                        distance_threshold_walk,
                        latent_random_walk,
                        momentum_random_walk,
                    )

                    if animation_type == "distance_threshold":
                        latent_frames, turn_around_step = distance_threshold_walk(
                            base_latent,
                            steps=anim_frames,
                            step_size=noise_strength,
                            explore_fraction=0.62,
                            seed=anim_seed,
                        )
                        st.info(
                            f"🔄 Distance threshold reached at step {turn_around_step}/{anim_frames}"
                        )
                    elif animation_type == "standard":
                        latent_frames = latent_random_walk(
                            base_latent,
                            steps=anim_frames,
                            step_size=noise_strength,
                            seed=anim_seed,
                        )
                    elif animation_type == "momentum":
                        latent_frames = momentum_random_walk(
                            base_latent,
                            steps=anim_frames,
                            step_size=noise_strength,
                            momentum=0.8,
                            seed=anim_seed,
                        )
                    elif animation_type == "attracted_home":
                        latent_frames = latent_random_walk(
                            base_latent,
                            steps=anim_frames,
                            step_size=noise_strength,
                            return_home=True,
                            seed=anim_seed,
                        )
                    elif animation_type == "deep_note":
                        print(f"🎵 Starting deep_note_walk with {anim_frames} frames")
                        latent_frames = deep_note_walk(
                            base_latent,
                            steps=anim_frames,
                            step_size=noise_strength,
                            seed=anim_seed,
                        )
                        print(
                            f"✅ deep_note_walk complete, got {len(latent_frames)} frames"
                        )

                    # Single progress update for walk generation (avoid per-frame sync overhead)
                    progress_bar.progress(0.5, text="Latent path generated")

                else:
                    # Generate breathing animation (sine, heartbeat, pulse)
                    latent_frames = breathing_animation(
                        base_latent,
                        frames=anim_frames,
                        max_strength=noise_strength,
                        pattern=animation_type,  # animation_type is now the pattern
                        seed=anim_seed,
                    )

                # Decode frames to images using batched processing
                progress_bar.progress(0.5, text="Decoding frames to images...")

                # Debug: Check if method exists
                print(
                    f"🔍 VAE model methods: {[m for m in dir(vae_model) if 'decode' in m]}"
                )
                print(f"🔍 Has decode_batch: {hasattr(vae_model, 'decode_batch')}")

                try:
                    print(
                        f"🔄 About to call decode_batch with {len(latent_frames)} frames"
                    )
                    image_frames = vae_model.decode_batch(latent_frames, batch_size=8)
                except Exception as e:
                    print(f"❌ Batch decode failed: {e}")
                    st.warning(f"Batch size 8 failed ({e}), trying smaller batches...")
                    image_frames = vae_model.decode_batch(latent_frames, batch_size=4)
                progress_bar.progress(1.0, text="Animation generation complete!")

                # Fuck PIL GIF encoder - use temp files + ffmpeg
                import subprocess
                import tempfile

                print(f"🎞️ Creating video: {len(image_frames)} images at {fps} fps")

                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save each frame as PNG
                    for i, img in enumerate(image_frames):
                        img.save(f"{temp_dir}/frame_{i:06d}.png")

                    # Use ffmpeg to create MP4
                    output_path = f"{temp_dir}/animation.mp4"
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        str(fps),
                        "-i",
                        f"{temp_dir}/frame_%06d.png",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-crf",
                        "18",  # High quality
                        output_path,
                    ]

                    print(f"🔄 Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        # Read MP4 into buffer
                        with open(output_path, "rb") as f:
                            gif_buffer = io.BytesIO(f.read())
                        print(
                            f"✅ MP4 created successfully ({len(gif_buffer.getvalue())} bytes)"
                        )
                    else:
                        print(f"❌ ffmpeg failed: {result.stderr}")
                        raise Exception(f"ffmpeg failed: {result.stderr}")

                st.success("✅ Animation generated!")

                # Embed MP4 video directly in Streamlit
                st.video(gif_buffer.getvalue(), format="video/mp4", start_time=0)

                # Download button
                st.download_button(
                    label="📥 Download Animation MP4",
                    data=gif_buffer.getvalue(),
                    file_name=f"latent_{animation_type}_{anim_frames}f_{fps}fps.mp4",
                    mime="video/mp4",
                    type="primary",
                )

            except Exception as e:
                st.error(f"❌ Error generating animation: {str(e)}")


def render_breathing_tab(vae_model: SD15VAE) -> None:
    """Render the latent breathing tab interface."""

    st.header("🌊 Latent Breathing Experiment")
    st.write(
        "Add rhythmic noise to explore latent space neighborhoods around an image."
    )

    # === PHASE 1: IMAGE SELECTION ===
    base_img = _handle_image_selection()
    if base_img is None:
        return

    # === PHASE 2: ENCODING ===
    st.subheader("🔄 Processing")

    with st.spinner("Encoding base image to latent space..."):
        try:
            base_latent = vae_model.encode(base_img)
            st.success("✅ Base image encoded successfully")
        except Exception as e:
            st.error(f"❌ Error encoding image: {str(e)}")
            return

    # === PHASE 3: BREATHING CONTROLS ===
    noise_strength, animation_type = _handle_breathing_controls()

    # === PHASE 4: PREVIEW ===
    st.subheader("🎯 Animation Preview")

    # Real-time preview controls
    preview_step, preview_seed, manual_t = _handle_preview_controls(animation_type)

    # Generate preview frame
    _generate_and_display_preview(
        vae_model,
        base_latent,
        animation_type,
        noise_strength,
        preview_step,
        preview_seed,
        manual_t,
    )

    # === PHASE 5: ANIMATION GENERATION ===
    _handle_animation_generation(vae_model, base_latent, animation_type, noise_strength)

    # === PHASE 6: EXPERIMENT NOTES ===
    with st.expander("🧪 Experiment Notes", expanded=False):
        st.write(
            """
        **Animation Types:**

        **Random Walks** (explore latent space):
        - **Distance Threshold**: Explores for 62% of frames, then SLERPs home
        - **Standard**: Pure random walk with no constraints
        - **Momentum**: Builds directional momentum for flowing paths
        - **Attracted Home**: Gradually biases back toward starting point
        - **Deep Note**: THX-inspired cinematic progression: distant start → meandering → dramatic SLERP arrival → hold

        **Breathing Patterns** (oscillate around original):
        - **Sine**: Smooth sinusoidal breathing rhythm
        - **Heartbeat**: Double-beat pulse pattern
        - **Pulse**: Sharp pulse breathing pattern

        **Breathing patterns:**
        - **Sine**: Smooth, meditative breathing
        - **Heartbeat**: Double-pulse like a heartbeat
        - **Pulse**: Sharp, rhythmic pulses

        **PCA Components:**
        - **PC0**: Usually captures the most significant variation (lighting, style)
        - **PC1-3**: Often control secondary features (pose, expression, color)
        - **PC4+**: Fine details and subtle variations

        **Artistic applications:**
        - **PCA (1.0-3.0)**: Coherent transformations following data manifold
        - **Random noise (0.1-2.0)**: Dream-like morphing effects
        - **High strength (3.0+)**: Abstract, surreal transformations
        """
        )
