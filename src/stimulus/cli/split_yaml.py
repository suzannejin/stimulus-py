#!/usr/bin/env python3
"""CLI module for splitting YAML configuration files into component configs.

This module provides functionality to split a single YAML configuration file into
separate component files: encoding config, individual split configs, and individual
transform configs. The resulting YAML files can be used independently.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from stimulus.data.interface.data_config_parser import ConfigDict, split_config_into_components

logger = logging.getLogger(__name__)


def split_yaml(config_yaml: str, out_dir_path: str) -> None:
    """Split a YAML config file into separate component configs.

    Takes a master YAML configuration and splits it into:
    - encode.yaml: Contains encoding configuration (global_params + columns)
    - split1.yaml, split2.yaml, etc.: Individual split configurations
    - transform1.yaml, transform2.yaml, etc.: Individual transform configurations
      with parameter expansion

    Args:
        config_yaml: Path to the master YAML configuration file.
        out_dir_path: Output directory to save the component config files.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the YAML config is invalid or malformed.
    """
    # Validate input file exists
    config_path = Path(config_yaml)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_yaml}")

    # Create output directory if it doesn't exist
    output_dir = Path(out_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate the YAML config
    yaml_config: dict[str, Any] = {}
    try:
        with open(config_yaml) as conf_file:
            yaml_config = yaml.safe_load(conf_file)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {config_yaml}: {e}") from e

    # Validate config structure
    try:
        config_dict = ConfigDict(**yaml_config)
    except Exception as e:
        raise ValueError(f"Invalid config structure in {config_yaml}: {e}") from e

    logger.info("YAML config loaded and validated successfully.")

    # Extract base name from config file path
    base_name = config_path.stem

    # Split the config into components
    split_config_into_components(config_dict, str(output_dir), base_name)

    # Count generated files for logging
    encoding_files = 1  # encode.yaml
    split_files = len(config_dict.split)
    transform_files = (
        sum(
            len(
                [
                    p
                    for p in transform.columns[0].transformations[0].params.values()
                    if isinstance(p, list) and len(p) > 1
                ],
            )
            if transform.columns
            and transform.columns[0].transformations
            and transform.columns[0].transformations[0].params
            else 1
            for transform in config_dict.transforms
        )
        if config_dict.transforms
        else 0
    )

    total_files = encoding_files + split_files + transform_files

    logger.info(f"Successfully generated {total_files} component configs:")
    logger.info(f"  - 1 encoding config ({base_name}_encode.yaml)")
    logger.info(f"  - {split_files} split configs ({base_name}_*_split.yaml)")
    logger.info(f"  - {transform_files} transform configs ({base_name}_*_transform.yaml)")
    logger.info(f"All files saved to: {out_dir_path}")
