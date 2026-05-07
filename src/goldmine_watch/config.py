"""Configuration loading and validation.

Loads the centralized YAML config file and exposes it as a typed object.
All pipeline stages should import config from this module rather than
parsing YAML directly.

Example:
    from goldmine_watch.config import load_config
    cfg = load_config("configs/mvp.yaml")
    print(cfg.geospatial.patch_size)
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class GeospatialConfig(BaseModel):
    """Geospatial settings with validation."""

    model_config = ConfigDict(extra="allow")
    patch_size: int = Field(..., gt=0, description="Patch size in pixels (must be positive)")


class DataConfig(BaseModel):
    """Data ingestion settings with validation."""

    model_config = ConfigDict(extra="allow")
    max_cloud_cover: float = Field(
        ...,
        ge=0,
        le=100,
        description="Maximum allowed cloud cover percentage (0-100)",
    )


class Config(BaseModel):
    """Root configuration model."""

    model_config = ConfigDict(extra="allow")
    geospatial: GeospatialConfig
    data: DataConfig


def load_config(config_path: str | Path) -> Config:
    """Load and validate the pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return Config.model_validate(raw)
