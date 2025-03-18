#!/usr/bin/env python3
"""CLI module for splitting CSV data files."""

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
        data_config_obj = data_config_parser.SplitConfigDict(**data_config_dict)

    splitters = data_config_parser.create_splitter(data_config_obj.split)
    transforms = data_config_parser.create_transforms(data_config_obj.transforms)
    split_columns = data_config_obj.split.split_input_columns

    return data_handlers.DatasetProcessor(
        csv_path=data_path,
        transforms=transforms,
        split_columns=split_columns,
        splitter=splitters,
    )


def split_csv(data_csv: str, config_yaml: str, out_path: str, *, force: bool = False) -> None:
    """Split the data according to the configuration.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output split CSV.
        force: Overwrite the split column if it already exists in the CSV.
    """
    # create a DatasetProcessor object from the config and the csv
    processor = load_data_config_from_path(data_csv, config_yaml)
    logger.info("Dataset processor initialized successfully.")

    # apply the split method to the data
    processor.add_split(force=force)
    logger.info("Split applied successfully.")

    # save the modified csv
    processor.save(out_path)
    logger.info("Split data saved successfully.")
