# GoldMine Watch — Agent Context

This file helps AI coding assistants understand the project and route to appropriate skills.

## Project

- **Name:** GoldMine Watch
- **Type:** Geospatial ML pipeline (Python)
- **Domain:** Satellite imagery semantic segmentation for environmental monitoring
- **Stack:** Python, PyTorch, GeoPandas, Rasterio, Dask, STAC

## Development Workflow

- **Package manager:** pip + pyproject.toml
- **Test runner:** pytest
- **Formatter:** black (line-length 100)
- **Linter:** ruff
- **Type checker:** mypy
- **Pre-commit:** enabled via `.pre-commit-config.yaml`
- **Pipeline orchestration:** Makefile

## Testing

```bash
# Run all tests
make test

# Run only unit tests
pytest tests/unit -m "not integration and not e2e"

# Run with coverage
pytest --cov=src/goldmine_watch --cov-report=term-missing
```

## Key Configuration

- Centralized config: `configs/mvp.yaml`
- All stages consume the same config file
- Never hardcode parameters in scripts

## Skill Routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
