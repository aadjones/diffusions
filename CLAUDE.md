# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

**Diffusion Art** - A latent space exploration toolkit for Stable Diffusion 1.5, focusing on interpolation between images encoded to latent space.

### Key Components

- **`src/diffusion_art/models/vae.py`**: SD15VAE class wrapping Stable Diffusion 1.5's VAE with scale factor 0.18215. Handles encoding/decoding between 512×512 images and 4-channel 64×64 latent tensors
- **`src/diffusion_art/core/interpolation.py`**: Core algorithms for SLERP (spherical linear interpolation) and LERP (linear interpolation) in latent space, plus multi-way blending
- **`app.py`**: Streamlit web interface for interactive latent space exploration with real-time interpolation

### Data Flow

Images → VAE Encode → Latent Space (4×64×64) → Interpolation → VAE Decode → Output Images

## Common Commands

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Running the Application
```bash
# Launch Streamlit web interface
streamlit run app.py
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/core/test_interpolation.py -v

# Run tests with coverage report
pytest --cov=src/diffusion_art --cov-report=html
```

### Code Quality
```bash
# Format code (black + isort)
make format

# Lint code
make lint

# Type checking
make typecheck
```

## Project Structure

```
src/diffusion_art/
├── core/
│   └── interpolation.py    # SLERP, LERP, multi-way interpolation
├── models/
│   └── vae.py             # SD15VAE wrapper
└── ui/                    # (Future UI components)

tests/unit/                # Unit tests mirroring src structure
app.py                     # Streamlit application entry point
```

## Technical Notes

- **Latent Space**: 4-channel tensors of size 64×64 representing 512×512 images (8× compression)
- **Device Support**: Auto-detects MPS (Apple Silicon) or falls back to CPU
- **Model Caching**: VAE model loading cached in Streamlit with `@st.cache_resource`
- **Image Processing**: Fixed 512×512 input size with LANCZOS resampling

## Streamlit Best Practices (2025)

### Image Display
- Use `width='stretch'` instead of deprecated `use_container_width=True`
- Use `width='content'` instead of deprecated `use_container_width=False`
- The `width` parameter replaced both `use_container_width` and `use_column_width` in 2025
- For responsive images that fill column width: `st.image(img, width='stretch')`
- For images at natural size: `st.image(img, width='content')`

### Performance
- Install `watchdog` for faster file watching and auto-reload
- Cache model loading with `@st.cache_resource` for expensive operations
- Optimize image sizes before upload for better performance

## Testing Philosophy

Focus unit tests on **core interpolation algorithms and edge cases**. Test mathematical properties (SLERP magnitude preservation, boundary conditions) rather than UI interactions or model loading.
