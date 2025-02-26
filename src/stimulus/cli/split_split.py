#!/usr/bin/env python3
"""CLI module for splitting YAML configuration files into unique files for each split.

This module provides functionality to split a single YAML configuration file into multiple
YAML files, each containing a unique split.
The resulting YAML files can be used as input configurations for the stimulus package.
"""

import logging
from typing import Any

import yaml

from stimulus.data.interface import data_config_parser

logger = logging.getLogger(__name__)


def split_split(config_yaml: str, out_dir_path: str) -> None:
    """Reads a YAML config file and generates a file per unique split.

    This script reads a YAML with a defined structure and creates all the YAML files ready to be passed to
    the stimulus package.

    The structure of the YAML is described here -> TODO paste here link to documentation.
    This YAML and its structure summarize how to generate unique splits and all the transformations associated to this split.

    This script will always generate at least one YAML file that represent the combination that does not touch the data (no transform)
    and uses the default split behavior.
    """
    # read the yaml experiment config and load its to dictionary
    yaml_config: dict[str, Any] = {}
    with open(config_yaml) as conf_file:
        yaml_config = yaml.safe_load(conf_file)

    yaml_config_dict = data_config_parser.ConfigDict(**yaml_config)

    logger.info("YAML config loaded successfully.")

    # generate the yaml files per split
    split_configs = data_config_parser.generate_split_configs(yaml_config_dict)

    logger.info("Splits generated successfully.")

    # dump all the YAML configs into files
    data_config_parser.dump_yaml_list_into_files(split_configs, out_dir_path, "test_split")

    logger.info("YAML files saved successfully.")
