.PHONY: install dev test lint format clean

# Lightweight install (no WildGuard / vLLM stack).
install:
	pip install -e .

# Add the optional WildGuard stack used for refusal eval + data prep.
install-wildguard:
	pip install -e ".[wildguard]"

# Dev tooling: pytest + ruff.
dev:
	pip install -e ".[dev]"

# Run the CPU-only unit test suite.
test:
	pytest -q

# Lint with ruff.
lint:
	ruff check .

# Auto-format with ruff.
format:
	ruff format .

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
