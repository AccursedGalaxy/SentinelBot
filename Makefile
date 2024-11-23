.PHONY: install test lint format clean pre-commit

install:
	pip install -r requirements.txt
	pip install pre-commit black flake8 pytest isort

test:
	pytest

lint:
	flake8 .
	black . --check
	isort . --check-only

format:
	black .
	isort .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +

pre-commit:
	pre-commit run --all-files
