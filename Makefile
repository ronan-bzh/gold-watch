# GoldMine Watch — Pipeline Orchestration
# Run `make help` to see available targets.

.PHONY: help setup install test lint format clean data train evaluate predict export

PYTHON := python3
PIP := $(PYTHON) -m pip
CONFIG := configs/mvp.yaml

help:
	@echo "GoldMine Watch — Available targets:"
	@echo "  make setup      Install dependencies and pre-commit hooks"
	@echo "  make data       Run data ingestion and preprocessing"
	@echo "  make train      Train the segmentation model"
	@echo "  make evaluate   Evaluate model on test set"
	@echo "  make predict    Run inference on pilot area"
	@echo "  make export     Export predictions to GeoTIFF and GeoPackage"
	@echo "  make test       Run the full test suite"
	@echo "  make lint       Run ruff and mypy"
	@echo "  make format     Run black and ruff --fix"
	@echo "  make clean      Remove processed artifacts and outputs"

setup: install
	pre-commit install
	@echo "Setup complete. Edit configs/mvp.yaml before running the pipeline."

install:
	$(PIP) install -e ".[dev]"

data:
	@echo "Running data ingestion and preprocessing..."
	$(PYTHON) -m goldmine_watch.data.ingest --config $(CONFIG)
	$(PYTHON) -m goldmine_watch.data.preprocess --config $(CONFIG)
	$(PYTHON) -m goldmine_watch.data.patches --config $(CONFIG)

train:
	@echo "Training segmentation model..."
	$(PYTHON) -m goldmine_watch.training.train --config $(CONFIG)

evaluate:
	@echo "Evaluating on test set..."
	$(PYTHON) -m goldmine_watch.training.evaluate --config $(CONFIG)

predict:
	@echo "Running inference on pilot area..."
	$(PYTHON) -m goldmine_watch.inference.predict --config $(CONFIG)

export:
	@echo "Exporting predictions..."
	$(PYTHON) -m goldmine_watch.export.raster --config $(CONFIG)
	$(PYTHON) -m goldmine_watch.export.vector --config $(CONFIG)

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

clean:
	rm -rf data/processed/*
	rm -rf data/splits/*
	rm -rf outputs/*
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
