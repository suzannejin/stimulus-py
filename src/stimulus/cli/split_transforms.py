#!/usr/bin/env python3
"""CLI module for splitting YAML configuration files into unique files for each transform.

This module provides functionality to split a single YAML configuration file into multiple
YAML files, each containing a unique transform associated to a unique split.
The resulting YAML files can be used as input configurations for the stimulus package.
"""

from typing import Any

import yaml

from stimulus.data.interface import data_config_parser


def split_transforms(config_yaml: str, out_dir_path: str) -> None:
    """Reads a YAML config and generates files for all split - transform possible combinations.

    This script reads a YAML with a defined structure and creates all the YAML files ready to be passed to the stimulus package.

    The structure of the YAML is described here -> TODO: paste here the link to documentation
    This YAML and its structure summarize how to generate all the transform for the split and respective parameter combinations.

    This script will always generate at least one YAML file that represent the combination that does not touch the data (no transform).
    """
    # read the yaml experiment config and load its dictionnary
    yaml_config: dict[str, Any] = {}
    with open(config_yaml) as conf_file:
        yaml_config = yaml.safe_load(conf_file)

    yaml_config_dict = data_config_parser.SplitConfigDict(**yaml_config)

    # Generate the yaml files for each transform
    split_transform_configs = data_config_parser.generate_split_transform_configs(yaml_config_dict)

    # Dump all the YAML configs into files
    data_config_parser.dump_yaml_list_into_files(split_transform_configs, out_dir_path, "test_transforms")
