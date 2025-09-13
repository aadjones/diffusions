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

### Device Management Gotchas

**⚠️ Critical:** All tensor operations must be on the same device. Common pitfalls:

- **✅ Correct:** `torch.randn(1, 4, 64, 64, device=base_tensor.device)`
- **❌ Wrong:** `torch.randn(1, 4, 64, 64)` (defaults to CPU)
- **✅ Correct:** `torch.linspace(0, 1, 64, device=base_tensor.device)`
- **❌ Wrong:** `torch.linspace(0, 1, 64)` (creates CPU tensor)

**Common sources of device mismatches:**
- PCA/sklearn operations (always return CPU tensors)
- New tensor creation without explicit device
- Constants and range tensors (`linspace`, `arange`, `zeros`, `ones`)
- Random operations (`randn`, `randperm`, `rand`)

**Fix:** Always specify `device=existing_tensor.device` when creating new tensors.

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

## Continuous Integration

The project includes GitHub Actions CI that runs on every push and PR:
- Uses Python 3.13 (matches local development)
- Runs `make test`, `make lint`, and `make typecheck`
- Ensures code quality before merging

## Type Safety Best Practices

**⚠️ Critical:** Always write type-safe code from the start to avoid mypy whack-a-mole. Follow these practices:

### Import Best Practices
```python
# ✅ Correct: Import specific types from typing
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

# ❌ Wrong: Generic containers without type parameters
from typing import Dict, List  # Then using Dict, List without [...]
```

### Function Signatures
```python
# ✅ Correct: Full type annotations
def process_latents(
    latents: List[torch.Tensor],
    config: AnimationConfig,
    batch_size: int = 8
) -> Tuple[List[Image.Image], AnimationMetrics]:

# ❌ Wrong: Missing annotations
def process_latents(latents, config, batch_size=8):
```

### Variable Declarations
```python
# ✅ Correct: Initialize with proper type
total_mse = 0.0  # float
results: Dict[str, Any] = {}
images: List[Image.Image] = []

# ❌ Wrong: Type inference issues
total_mse = 0  # int, causes issues later
results = {}  # mypy can't infer type
```

### Collection Handling
```python
# ✅ Correct: Type-safe list operations
recommendations: List[str] = []
recommendations.append("suggestion")

# ❌ Wrong: Dynamic typing that breaks mypy
analysis["recommendations"] = []  # mypy sees this as Collection[str]
analysis["recommendations"].append("suggestion")  # error!
```

### Exception Handling
```python
# ✅ Correct: Specific exception types
try:
    result = risky_operation()
except ValueError as e:
    handle_error(e)
except Exception as e:  # Fallback for unknown errors
    log_error(e)

# ❌ Wrong: Bare except
try:
    result = risky_operation()
except:  # Catches everything, hard to debug
    pass
```

### Method Assignment (Monkey Patching)
```python
# ✅ Correct: Use type ignore for dynamic method replacement
def new_method(self, x: int) -> str:
    return str(x)

obj.method = new_method  # type: ignore[method-assign]

# ❌ Wrong: No type ignore causes mypy error
obj.method = new_method  # mypy error
```

### Return Type Consistency
```python
# ✅ Correct: Consistent return types
def get_results(self) -> Dict[str, Dict[str, float]]:
    if not self.data:
        return {"error": {"value": 0.0, "message": "No data"}}
    return {"results": {"accuracy": 0.95, "loss": 0.05}}

# ❌ Wrong: Inconsistent return types
def get_results(self) -> Dict[str, float]:  # Wrong annotation
    if not self.data:
        return {"error": "No data"}  # str, not float
    return {"accuracy": 0.95}  # float
```

### Common Patterns
```python
# ✅ Correct: Optional handling
def safe_decode(latent: Optional[torch.Tensor]) -> Optional[Image.Image]:
    if latent is None:
        return None
    return decode(latent)

# ✅ Correct: Union types for multiple return formats
def flexible_parse(data: str) -> Union[dict, str]:
    try:
        return json.loads(data)
    except ValueError:
        return f"Invalid JSON: {data}"
```

### Performance-Critical Code
```python
# ✅ Correct: Type ignore for performance hacks that mypy doesn't understand
def optimized_decode(self, latents: List[torch.Tensor]) -> List[Image.Image]:
    # Use dynamic method replacement for performance
    original_method = self.decode_method
    self.decode_method = fast_decode_method  # type: ignore[method-assign]

    try:
        return self.process_batch(latents)
    finally:
        self.decode_method = original_method  # type: ignore[method-assign]
```

**Rule of thumb:** If mypy complains, fix the types at the source rather than adding `# type: ignore` unless it's truly necessary for dynamic behavior or performance optimizations.

## Testing Philosophy

Focus unit tests on **core interpolation algorithms and edge cases**. Test mathematical properties (SLERP magnitude preservation, boundary conditions) rather than UI interactions or model loading.
