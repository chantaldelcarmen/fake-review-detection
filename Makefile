.PHONY: help install install-dev clean lint format test train evaluate

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code with black and isort"
	@echo "  make test         - Run tests"
	@echo "  make train        - Train the model"
	@echo "  make evaluate     - Evaluate the model"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf outputs/ multirun/ .hydra/

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/ --line-length=100
	isort src/ tests/ scripts/ --profile black

test:
	pytest tests/ -v --cov=src/fake_review_detection --cov-report=html --cov-report=term

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

preprocess:
	python scripts/preprocess_data.py

feature-ablation:
	python scripts/feature_ablation.py
