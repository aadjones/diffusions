# Diffusion Art - Latent Space Exploration

A toolkit for exploring and manipulating latent space representations in Stable Diffusion models, focusing on artistic applications through interpolation and visualization.

## Features

- **VAE Encoding/Decoding**: Encode images to SD 1.5 latent space and decode back to images
- **Spherical Linear Interpolation (SLERP)**: Smooth transitions between latent representations
- **Linear Interpolation (LERP)**: Alternative interpolation method for comparison
- **Multi-way Interpolation**: Blend multiple latent vectors with custom weights
- **Streamlit Web UI**: Interactive interface for real-time latent exploration
- **Comprehensive Testing**: Unit tests for core algorithms and edge cases

## Installation

### Prerequisites

- Python 3.13 (recommended for best performance)
- macOS with Apple Silicon (MPS support) or x86_64 with CUDA/CPU

### Setup

1. Clone the repository:

```bash
git clone https://github.com/aadjones/diffusions
cd diffusions
```

2. Set up development environment:

```bash
make setup    # Create virtual environment
make install  # Install all dependencies
```

Alternatively, manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Web Interface

Launch the Streamlit app for interactive exploration:

```bash
make run  # Recommended
# or
streamlit run app.py
```

This opens a web interface where you can:

- Upload two images
- Adjust the interpolation slider to morph between them
- See real-time results in the latent space

### Programmatic Usage

```python
from src.diffusion_art.models.vae import SD15VAE
from src.diffusion_art.core.interpolation import slerp, lerp
from PIL import Image

# Initialize VAE
vae = SD15VAE()

# Load and encode images
img1 = Image.open("image1.jpg")
img2 = Image.open("image2.jpg")

z1 = vae.encode(img1)
z2 = vae.encode(img2)

# Interpolate in latent space
interpolated = slerp(z1, z2, t=0.5)  # Midpoint

# Decode back to image
result_image = vae.decode(interpolated)
result_image.save("interpolated.jpg")
```

### Advanced Usage

```python
from src.diffusion_art.core.interpolation import multi_slerp, create_interpolation_path

# Multi-way interpolation
latents = [z1, z2, z3, z4]
weights = [0.3, 0.2, 0.3, 0.2]
blended = multi_slerp(latents, weights)

# Create interpolation path
path = create_interpolation_path(z1, z2, steps=10, method="slerp")
for i, latent in enumerate(path):
    frame = vae.decode(latent)
    frame.save(f"frame_{i:03d}.jpg")
```

## Architecture

```
src/diffusion_art/
├── core/
│   └── interpolation.py    # Interpolation algorithms
├── models/
│   └── vae.py             # VAE wrapper for SD 1.5
└── ui/                    # Future UI components
```

### Key Components

- **SD15VAE**: Wrapper around Stable Diffusion 1.5's VAE with proper scaling
- **SLERP**: Spherical linear interpolation preserving angular relationships
- **LERP**: Standard linear interpolation for baseline comparison
- **Multi-SLERP**: N-way interpolation with weighted blending

## Development

The project includes a comprehensive Makefile for development tasks:

```bash
# Development workflow
make format     # Format code (black + isort + pre-commit hooks)
make lint       # Lint code (flake8)
make typecheck  # Type checking (mypy)
make test       # Run tests
make clean      # Clean up cache files
```

### Running Tests

```bash
# Using Makefile (recommended)
make test

# Direct pytest usage
pytest
pytest --cov=src/diffusion_art                    # With coverage
pytest tests/unit/core/test_interpolation.py -v   # Specific test
```

### Project Structure

```
diffusion/
├── src/diffusion_art/          # Main package
├── tests/                      # Test suite
├── app.py                      # Streamlit application
├── requirements.txt            # Dependencies
├── Makefile                    # Development commands
├── pyproject.toml             # Project configuration
├── .pre-commit-config.yaml    # Pre-commit hooks
├── .flake8                    # Flake8 configuration
├── mypy.ini                   # MyPy configuration
├── .github/workflows/ci.yml   # CI/CD pipeline
└── CLAUDE.md                  # AI assistant instructions
```

## Technical Details

### Latent Space

Stable Diffusion 1.5 uses a 4-channel latent space of size 64×64, representing 512×512 pixel images with a compression factor of ~8×. The VAE scale factor of 0.18215 is applied during encoding/decoding.

### Interpolation Methods

- **SLERP**: Maintains constant magnitude while interpolating along great circles on the unit sphere. Better for preserving semantic meaning in high-dimensional spaces.
- **LERP**: Direct linear interpolation in Euclidean space. Simpler but may produce less semantically coherent results.

### Performance

- Model loading is cached using Streamlit's `@st.cache_resource`
- Supports MPS (Apple Silicon) and CPU backends
- Includes `watchdog` for faster file watching and auto-reload
- Batch processing capabilities for generating sequences

## Limitations

- Currently supports SD 1.5 VAE only
- Requires significant VRAM/memory for larger models
- Image preprocessing fixes input size to 512×512

## Future Enhancements

- Support for other diffusion model VAEs
- Advanced path planning (geodesics, loops)
- Real-time audio-driven interpolation
- Fisher information metric integration
- PCA/UMAP visualization of latent spaces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stable Diffusion by Stability AI
- Diffusers library by Hugging Face
