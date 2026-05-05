# TODOS

This file captures deferred work that was considered during the engineering review but not included in the immediate scope.

## TODO-1: Evaluate TorchGeo for geospatial data loading and inference

- **What:** Evaluate [TorchGeo](https://torchgeo.readthedocs.io/) as a replacement for custom patch generation, dataset splitting, and sliding-window inference utilities.
- **Why:** The pipeline currently plans custom rasterio-based patch extraction and manual sliding-window inference. TorchGeo is a Layer 1 proven technology for geospatial deep learning. It provides `GeoDataset`, `RandomGeoSampler`, and inference utilities that handle CRS and spatial transforms correctly. Using it could eliminate hundreds of lines of custom code and reduce the risk of subtle geospatial transform bugs.
- **Pros:**
  - Reduces custom code volume and maintenance burden.
  - Well-tested, community-maintained geospatial abstractions.
  - Native support for patch-based datasets and sliding-window inference.
- **Cons:**
  - Adds a dependency.
  - May require API changes if TorchGeo doesn't support a specific feature (e.g., custom cloud masking strategy).
- **Context:** The pipeline currently plans custom `rasterio.windows`-based patch extraction and a manual tiling/blending loop for inference. TorchGeo's `IntersectionDataset` and `RandomGeoSampler` could replace much of this. If TorchGeo is missing a needed feature, the fallback is to keep the custom implementation.
- **Depends on:** Milestone 2 (dataset preparation) and Milestone 5 (inference). Can be evaluated in parallel with Milestone 1.
- **Priority:** Medium
- **Status:** Not started

## TODO-2: Set up pre-commit hooks with black, ruff, and mypy

- **What:** Add `.pre-commit-config.yaml` with black (formatting), ruff (linting), and mypy (type checking) hooks.
- **Why:** The user explicitly prefers explicit over clever and well-tested code. Automated formatting and linting enforce consistency, catch type errors before review, and keep the codebase maintainable. Retrofitting these tools later requires a massive "format everything" commit that pollutes git history.
- **Pros:**
  - Enforces consistent style across all contributions.
  - Catches type errors and common bugs before they reach CI.
  - Cheap to set up now (5 minutes), expensive to retrofit later.
- **Cons:**
  - Occasional `--no-verify` needed for WIP commits.
  - Slight initial configuration overhead.
- **Context:** Can be added immediately after the initial repo setup (`pyproject.toml`, `src/` layout). Runs on every commit.
- **Depends on:** Python package structure (src/goldmine_watch/).
- **Priority:** Medium
- **Status:** In progress — .pre-commit-config.yaml created, needs `pre-commit install`

## TODO-3: Set up CI/CD for tests

- **What:** Add a GitHub Actions workflow (or equivalent) that runs `make test` on every push and pull request.
- **Why:** Ensures that contributions don't break existing functionality. Essential for a multi-stage pipeline where a change in data loading can silently break training.
- **Priority:** Low (post-MVP)
- **Status:** Not started

## TODO-4: Evaluate PyTorch Lightning for training loop boilerplate

- **What:** Evaluate PyTorch Lightning as a replacement for the custom training loop.
- **Why:** Lightning handles checkpointing, logging, early stopping, and distributed training with minimal boilerplate. Could reduce `training/` module from ~300 lines to ~50 lines.
- **Priority:** Low (post-MVP)
- **Status:** Not started

## TODO-5: Add DVC for data and model versioning

- **What:** Integrate [DVC](https://dvc.org/) to version-control large artifacts (raw imagery, composites, checkpoints) alongside git.
- **Why:** The `.gitignore` approach keeps large files out of git but provides no versioning. DVC enables reproducible experiments by tracking exact data versions.
- **Priority:** Low (post-MVP)
- **Status:** Not started
