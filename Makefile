.PHONY: clean setup install format lint typecheck run test docs docs-clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  clean     - Remove cache files and build artifacts"
	@echo "  setup     - Create virtual environment"
	@echo "  install   - Install dependencies"
	@echo "  format    - Format code with black and isort"
	@echo "  lint      - Lint code with flake8"
	@echo "  typecheck - Type check with mypy"
	@echo "  run       - Run the Streamlit app"
	@echo "  test      - Run tests with pytest"
	@echo "  docs      - Build documentation with Sphinx"
	@echo "  docs-clean - Clean documentation build files"

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	@echo "Clean complete."

setup:
	@echo "Setting up virtual environment..."
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e ".[dev]"
	@echo "Dependencies installed."

format:
	@echo "Formatting code..."
	pre-commit run --all-files
	@echo "Code formatting complete."

lint:
	@echo "Linting code..."
	flake8 src/ tests/ app.py
	@echo "Linting complete."

typecheck:
	@echo "Type checking..."
	mypy src/ app.py
	@echo "Type checking complete."

run:
	@echo "Starting Streamlit app..."
	streamlit run app.py

test:
	@echo "Running tests..."
	pytest
	@echo "Tests complete."

docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "Documentation built at docs/_build/html/index.html"

docs-clean:
	@echo "Cleaning documentation build files..."
	cd docs && make clean
	@echo "Documentation clean complete."
