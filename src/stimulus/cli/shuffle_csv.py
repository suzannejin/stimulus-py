#!/usr/bin/env python3
"""CLI module for shuffling CSV data files."""

import logging

import yaml

from stimulus.data import data_handlers
from stimulus.data.interface import data_config_parser

logger = logging.getLogger(__name__)


def load_data_config_from_path(data_path: str, data_config_path: str) -> data_handlers.DatasetProcessor:
    """Load the data config from a path.

    Args:
        data_config_path: Path to the data config file.

    Returns:
        A tuple of the parsed configuration.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.SplitConfigDict(**data_config_dict)

    splitters = data_config_parser.create_splitter(data_config_obj.split)
    transforms = data_config_parser.create_transforms(data_config_obj.transforms)
    split_columns = data_config_obj.split.split_input_columns
    label_columns = [column.column_name for column in data_config_obj.columns if column.column_type == "label"]

    return data_handlers.DatasetProcessor(
        csv_path=data_path,
        transforms=transforms,
        split_columns=split_columns,
        splitter=splitters,
    ), label_columns


def shuffle_csv(data_csv: str, config_yaml: str, out_path: str) -> None:
    """Shuffle the data and split it according to the default split method.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output shuffled CSV.
    """
    # create a DatasetProcessor object from the config and the csv
    processor, label_columns = load_data_config_from_path(data_csv, config_yaml)
    logger.info("Dataset processor initialized successfully.")

    # shuffle the data with a default seed
    # TODO: get the seed from the config if and when that is going to be set there
    processor.shuffle_labels(label_columns, seed=42)
    logger.info("Data shuffled successfully.")

    # save the modified csv
    processor.save(out_path)
    logger.info("Shuffled data saved successfully.")
