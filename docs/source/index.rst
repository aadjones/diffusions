.. Diffusion Art documentation master file, created by
   sphinx-quickstart on Sat Sep 13 13:44:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Diffusion Art Documentation
===========================

Diffusion Art is a latent space exploration toolkit for Stable Diffusion 1.5,
focusing on interpolation between images encoded to latent space.

Features
--------

- **Latent Space Encoding/Decoding**: Convert 512×512 images to/from 4-channel 64×64 latent tensors
- **Interpolation Algorithms**: SLERP (spherical linear interpolation) and LERP (linear interpolation)
- **Interactive Web Interface**: Real-time exploration via Streamlit
- **Multi-way Blending**: Advanced interpolation between multiple images
- **Device Optimization**: Auto-detects MPS (Apple Silicon) or CPU

Quick Start
-----------

1. Install the package:

.. code-block:: bash

   pip install -e ".[dev]"

2. Launch the web interface:

.. code-block:: bash

   streamlit run app.py

3. Upload images and explore latent space interpolations!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
