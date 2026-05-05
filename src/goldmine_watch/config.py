"""Configuration loading and validation.

Loads the centralized YAML config file and exposes it as a typed object.
All pipeline stages should import config from this module rather than
parsing YAML directly.

Example:
    from goldmine_watch.config import load_config
    cfg = load_config("configs/mvp.yaml")
    print(cfg.model.architecture)
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all pipeline parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    # TODO: Add pydantic or OmegaConf validation here
    return cfg
