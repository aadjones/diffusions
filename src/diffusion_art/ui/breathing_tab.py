"""Refactored latent breathing tab using controller pattern."""

import asyncio
import io
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

from ..models.vae import SD15VAE
from ..utils import presets
from .controllers import BreathingTabController


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
    controller: BreathingTabController,
    animation_type: str,
    noise_strength: float,
    preview_step: Optional[int],
    preview_seed: int,
    manual_t: Optional[float],
) -> None:
    """Generate and display preview frame using controller."""
    with st.spinner("Generating preview..."):
        try:
            # Create animation config
            config = controller.create_animation_config(
                animation_type=animation_type,
                frames=31,  # Preview frames
                fps=24,
                noise_strength=noise_strength,
                seed=preview_seed,
            )

            # Determine frame index based on animation type
            if animation_type in ["sine", "heartbeat", "pulse"]:
                # For breathing patterns, use manual_t to calculate frame index
                if manual_t is not None:
                    frame_index = int(manual_t * 30)  # 0-30 frames
                else:
                    frame_index = 0
            else:
                # For random walks, use preview_step
                frame_index = preview_step if preview_step is not None else 0

            # Generate preview
            preview_img, error, metadata = controller.generate_preview(
                config, frame_index
            )

            if error:
                st.error(f"❌ Error generating preview: {error}")
                return

            if preview_img is None:
                st.error("❌ No preview image generated")
                return

            # Display preview with appropriate caption
            if animation_type in ["sine", "heartbeat", "pulse"]:
                t_display = manual_t if manual_t is not None else 0.0
                caption = (
                    f"🌊 {animation_type.title()} Breathing Preview (t={t_display:.2f})"
                )
            else:
                caption = (
                    f"🚶 Random Walk Preview (Step {frame_index}, {animation_type})"
                )

                # Add turn around info for distance threshold walks
                if (
                    animation_type == "distance_threshold"
                    and metadata
                    and metadata.get("turn_around_step")
                ):
                    turn_step = metadata["turn_around_step"]
                    if frame_index <= turn_step:
                        caption += " [Exploring]"
                    else:
                        caption += " [Returning home via SLERP]"
                    st.write(f"🔄 Turned around at step {turn_step}")

            st.image(preview_img, caption=caption, width="stretch")

            # Show performance info if available
            if metadata:
                if metadata.get("generation_time"):
                    st.caption(f"⚡ Generated in {metadata['generation_time']:.2f}s")

        except Exception as e:
            st.error(f"❌ Error generating preview: {str(e)}")


async def _handle_animation_generation_async(
    controller: BreathingTabController,
    animation_type: str,
    noise_strength: float,
    duration_seconds: float,
    fps: int,
    anim_seed: int,
    keyframe_interval: int,
) -> None:
    """Handle animation generation asynchronously."""
    anim_frames = int(duration_seconds * fps)

    # Create config
    config = controller.create_animation_config(
        animation_type=animation_type,
        frames=anim_frames,
        fps=fps,
        noise_strength=noise_strength,
        seed=anim_seed,
    )

    # Create progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def progress_callback(progress: float, message: str) -> None:
        progress_bar.progress(progress)
        progress_text.text(message)

    try:
        # Show keyframe optimization info
        if keyframe_interval > 1:
            quality_info = {
                8: "Fast mode (5x speedup)",
                4: "Balanced mode (3x speedup)",
            }
            st.info(
                f"⚡ Using {quality_info.get(keyframe_interval, 'keyframe optimization')}"
            )

        # Generate animation asynchronously
        video_bytes, error, metrics = await controller.generate_animation_async(
            config, keyframe_interval, progress_callback
        )

        if error:
            st.error(f"❌ Error generating animation: {error}")
            return

        if video_bytes is None:
            st.error("❌ No video generated")
            return

        progress_bar.progress(1.0)
        progress_text.text("Animation generation complete!")

        st.success("✅ Animation generated!")

        # Display metrics
        if metrics:
            st.info(
                f"📊 Generated {metrics['total_frames']} frames "
                f"in {metrics['generation_time_seconds']:.1f}s "
                f"(Peak memory: {metrics['peak_memory_mb']:.1f}MB)"
            )

            if metrics.get("turn_around_step"):
                st.info(
                    f"🔄 Distance threshold reached at step {metrics['turn_around_step']}/{metrics['total_frames']}"
                )

        # Embed video
        st.video(video_bytes, format="video/mp4", start_time=0)

        # Download button
        st.download_button(
            label="📥 Download Animation MP4",
            data=video_bytes,
            file_name=f"latent_{animation_type}_{anim_frames}f_{fps}fps.mp4",
            mime="video/mp4",
            type="primary",
        )

    except Exception as e:
        st.error(f"❌ Error generating animation: {str(e)}")
    finally:
        progress_bar.empty()
        progress_text.empty()


def _handle_animation_generation(
    controller: BreathingTabController,
    animation_type: str,
    noise_strength: float,
) -> None:
    """Handle animation generation UI and processing."""
    st.subheader("🎬 Generate Animation")

    # Animation controls
    col6a, col6b, col8 = st.columns(3)
    with col6a:
        duration_seconds = st.slider(
            "Duration (seconds)", 1.0, 30.0, 7.0, 0.5, help="Total animation duration"
        )
    with col6b:
        fps = st.slider(
            "Frame Rate",
            10,
            30,
            24,
            1,
            help="Frames per second (capped at 30 for performance)",
        )
    with col8:
        anim_seed = st.number_input(
            "Animation Seed", value=42, min_value=0, max_value=9999
        )

    # Calculate and display frame info
    anim_frames = int(duration_seconds * fps)
    frame_duration = round(1000 / fps)

    # Performance Controls
    st.subheader("⚡ Performance Settings")

    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        quality_mode = st.selectbox(
            "Quality Mode",
            options=["Fast (5x speedup)", "Balanced (3x speedup)", "Full Quality"],
            index=0,  # Default to Fast
            help="Fast mode renders keyframes and interpolates between them for major speed gains",
        )

    with perf_col2:
        # Map quality modes to keyframe intervals
        keyframe_map = {
            "Fast (5x speedup)": 8,
            "Balanced (3x speedup)": 4,
            "Full Quality": 1,
        }
        keyframe_interval = keyframe_map[quality_mode]

        # Show estimated render time
        frames_to_decode = len(range(0, anim_frames, keyframe_interval))
        if (anim_frames - 1) % keyframe_interval != 0:
            frames_to_decode += 1

        decode_time = frames_to_decode * 2  # 2 seconds per frame from benchmarks
        st.metric(
            label="Estimated Render Time",
            value=f"{decode_time // 60:.0f}m {decode_time % 60:.0f}s",
            delta=f"Decoding {frames_to_decode}/{anim_frames} frames",
        )

    st.info(f"📊 {anim_frames} frames at {fps} fps ({frame_duration}ms per frame)")

    if st.button("🎬 Generate Animation", type="primary", key="generate_animation_btn"):
        # Run async animation generation
        asyncio.run(
            _handle_animation_generation_async(
                controller,
                animation_type,
                noise_strength,
                duration_seconds,
                fps,
                anim_seed,
                keyframe_interval,
            )
        )


def render_breathing_tab(vae_model: SD15VAE) -> None:
    """Render the refactored latent breathing tab interface."""

    st.header("🌊 Latent Breathing Experiment")
    st.write(
        "Add rhythmic noise to explore latent space neighborhoods around an image."
    )

    # Initialize controller
    if "breathing_controller" not in st.session_state:
        st.session_state.breathing_controller = BreathingTabController(vae_model)

    controller = st.session_state.breathing_controller

    # === PHASE 1: IMAGE SELECTION ===
    base_img = _handle_image_selection()
    if base_img is None:
        return

    # === PHASE 2: ENCODING ===
    st.subheader("🔄 Processing")

    with st.spinner("Encoding base image to latent space..."):
        if controller.encode_base_image(base_img):
            st.success("✅ Base image encoded successfully")
        else:
            st.error("❌ Error encoding image")
            return

    # === PHASE 3: BREATHING CONTROLS ===
    noise_strength, animation_type = _handle_breathing_controls()

    # === PHASE 4: PREVIEW ===
    st.subheader("🎯 Animation Preview")

    # Real-time preview controls
    preview_step, preview_seed, manual_t = _handle_preview_controls(animation_type)

    # Generate preview frame
    _generate_and_display_preview(
        controller,
        animation_type,
        noise_strength,
        preview_step,
        preview_seed,
        manual_t,
    )

    # === PHASE 5: ANIMATION GENERATION ===
    _handle_animation_generation(controller, animation_type, noise_strength)

    # === PHASE 6: CACHE MANAGEMENT ===
    with st.expander("⚡ Performance", expanded=False):
        col_stats, col_clear = st.columns([2, 1])

        with col_stats:
            cache_stats = controller.get_cache_stats()
            st.write("**Cache Status:**")
            st.write(f"- Cached sequences: {cache_stats['cached_sequences']}")
            st.write(f"- Total cached tensors: {cache_stats['total_cached_tensors']}")
            st.write(f"- Cache limit: {cache_stats['cache_size_limit']}")

        with col_clear:
            if st.button(
                "🧹 Clear Cache", help="Free up memory by clearing cached previews"
            ):
                controller.clear_cache()
                st.success("Cache cleared!")
                st.rerun()

    # === PHASE 7: EXPERIMENT NOTES ===
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

        **Artistic applications:**
        - **Low strength (0.1-2.0)**: Subtle, dream-like morphing effects
        - **Medium strength (1.0-3.0)**: Coherent transformations following data manifold
        - **High strength (3.0+)**: Abstract, surreal transformations

        **Performance Modes:**
        - **Fast (5x speedup)**: Keyframe interval=8, renders only 1/8th of frames
        - **Balanced (3x speedup)**: Keyframe interval=4, renders 1/4th of frames
        - **Full Quality**: Renders every frame (slower but highest fidelity)

        *Keyframe modes interpolate missing frames in image space for major speed gains*
        """
        )
