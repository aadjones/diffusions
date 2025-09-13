Installation
============

Requirements
------------

- Python 3.8 or higher
- PyTorch 2.0 or higher
- 4GB+ RAM recommended
- Apple Silicon (MPS) support available

Basic Installation
------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/user/diffusion-art.git
   cd diffusion-art

2. Create a virtual environment:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install the package:

.. code-block:: bash

   pip install -e .

Development Installation
------------------------

For development with testing and linting tools:

.. code-block:: bash

   pip install -e ".[dev]"

Additional Dependencies
-----------------------

For analysis features:

.. code-block:: bash

   pip install -e ".[analysis]"

For Jupyter notebook support:

.. code-block:: bash

   pip install -e ".[notebooks]"

For documentation building:

.. code-block:: bash

   pip install -e ".[docs]"

Verification
------------

Test your installation:

.. code-block:: bash

   python -c "import diffusion_art; print('Installation successful!')"

Run the test suite:

.. code-block:: bash

   pytest

Launch the web interface:

.. code-block:: bash

   streamlit run app.py
