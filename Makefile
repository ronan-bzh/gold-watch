# GoldMine Watch — Pipeline Orchestration
# Run `make help` to see available targets.

.PHONY: help setup install test lint format clean docker-build docker-run docker-compose-up

PYTHON := python3
PIP := $(PYTHON) -m pip
CONFIG := configs/mvp.yaml

help:
	@echo "GoldMine Watch — Available targets:"
	@echo "  make setup      Install dependencies and pre-commit hooks"
	@echo "  make install    Install package in editable mode"
	@echo "  make test       Run the full test suite"
	@echo "  make lint       Run ruff and mypy"
	@echo "  make format     Run black and ruff --fix"
	@echo "  make clean      Remove processed artifacts and outputs"
	@echo ""
	@echo "Docker deployment:"
	@echo "  make docker-build       Build the Docker image"
	@echo "  make docker-run         Run the container on port 8000"
	@echo "  make docker-compose-up  Start via Docker Compose"
	@echo ""
	@echo "Pipeline stages (run manually):"
	@echo "  python scripts/plot_labels.py data/raw/labels.gpkg"
	@echo "  python scripts/download_one_scene.py"
	@echo "  python scripts/make_patches.py <image.tif> <labels.gpkg>"
	@echo "  python -m goldmine_watch.training.train --fake <img> <labels>"
	@echo "  python -m goldmine_watch.training.train --patches <dir>"
	@echo "  python -m goldmine_watch.inference.predict <patch.npy> <model.pth>"
	@echo "  python -m goldmine_watch.inference.predict_big <image.tif> <model.pth>"

setup: install
	pre-commit install
	@echo "Setup complete. Edit configs/mvp.yaml before running the pipeline."

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/unit -v --tb=short

lint:
	ruff check src tests scripts
	mypy src

format:
	black src tests scripts
	ruff check --fix src tests scripts

clean:
	rm -rf data/processed/*
	rm -rf data/splits/*
	rm -rf outputs/*
	rm -rf models/*
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t goldmine-watch ./web

docker-run:
	docker run -p 8000:8000 goldmine-watch

docker-compose-up:
	docker compose -f web/docker-compose.yml up -d
