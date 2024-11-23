.PHONY: install test lint format clean pre-commit security-check coverage

# Variables
PYTHON = python3
PIP = pip
PYTEST = pytest
COVERAGE = coverage
RUFF = ruff
BLACK = black
ISORT = isort
MYPY = mypy
BANDIT = bandit

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pre-commit black isort mypy ruff bandit pytest coverage
	pre-commit install

test:
	$(PYTHON) -m pytest -v

coverage:
	$(PYTHON) -m coverage run -m pytest
	$(PYTHON) -m coverage report
	$(PYTHON) -m coverage html

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check-only .
	$(PYTHON) -m mypy .

format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
	$(PYTHON) -m ruff format .

security-check:
	$(PYTHON) -m bandit -r .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +

pre-commit:
	pre-commit run --all-files
