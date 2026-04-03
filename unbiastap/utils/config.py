import logging
from pathlib import Path
from typing import Union

import yaml

logger = logging.getLogger(__name__)

def load_config(path: Union[str, Path]) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path.resolve()}'"
        )

    if config_path.suffix not in {".yml", ".yaml"}:
        raise ValueError(
            f"Expected a YAML file (.yml or .yaml), got '{config_path.suffix}'"
        )

    logger.debug("Loading config from '%s'", config_path.resolve())

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected config file to parse to a dict, got "
            f"{type(config).__name__}"
        )

    logger.debug("Config loaded with top-level keys: %s", list(config.keys()))
    return config
