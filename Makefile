.PHONY: help install dev test lint format clean docs build

help:
	@echo "Available commands:"
	@echo "  install    Install package in production mode"
	@echo "  dev        Install package in development mode with all dependencies"
	@echo "  test       Run tests without coverage"
	@echo "  test-cov   Run tests with coverage"
	@echo "  lint       Run linting checks (ruff, mypy, pydoclint)"
	@echo "  format     Format code with ruff"
	@echo "  clean      Remove build artifacts and cache files"
	@echo "  docs       Build documentation"
	@echo "  build      Build distribution packages"

install:
	uv pip install .

dev:
	uv sync --all-groups
	uv run pre-commit install

test:
	uv run pytest

test-cov:
	uv run pytest --cov=naampy --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check .
	uv run pyright
	uv run pydoclint naampy/

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

docs:
	uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html

build:
	uv build
