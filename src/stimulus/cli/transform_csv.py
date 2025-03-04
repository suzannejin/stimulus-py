#!/usr/bin/env python3
"""CLI module for transforming CSV data files."""

import logging

import yaml

from stimulus.data import data_handlers
from stimulus.data.interface import data_config_parser

logger = logging.getLogger(__name__)


def load_data_config_from_path(data_path: str, data_config_path: str) -> data_handlers.DatasetProcessor:
    """Load the data config from a path.

    Args:
        data_path: Path to the data file.
        data_config_path: Path to the data config file.

    Returns:
        A DatasetProcessor instance configured with the data.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.SplitTransformDict(**data_config_dict)

    transforms = data_config_parser.create_transforms([data_config_obj.transforms])
    splitter = data_config_parser.create_splitter(data_config_obj.split)
    split_columns = data_config_obj.split.split_input_columns

    return data_handlers.DatasetProcessor(
        csv_path=data_path,
        transforms=transforms,
        split_columns=split_columns,
        splitter=splitter,
    )


def main(data_csv: str, config_yaml: str, out_path: str) -> None:
    """Transform the data according to the configuration.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output transformed CSV.
    """
    # Create a DatasetProcessor object from the config and the csv
    processor = load_data_config_from_path(data_csv, config_yaml)
    logger.info("Dataset processor initialized successfully.")

    # Apply the transformations to the data
    processor.apply_transformations()
    logger.info("Transformations applied successfully.")

    # Save the modified csv
    processor.save(out_path)
    logger.info("Transformed data saved successfully.")
