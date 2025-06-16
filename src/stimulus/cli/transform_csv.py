#!/usr/bin/env python3
"""CLI module for transforming CSV data files."""

import logging
import os
from typing import Any

import datasets
import pandas as pd
import pyarrow as pa
import yaml

from stimulus.data.interface import data_config_parser

logger = logging.getLogger(__name__)


def load_transforms_from_config(data_config_path: str) -> dict[str, list[Any]]:
    """Load the data config from a path.

    Args:
        data_config_path: Path to the data config file.

    Returns:
        A DatasetProcessor instance configured with the data.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.SplitTransformDict(**data_config_dict)

    return data_config_parser.create_transforms([data_config_obj.transforms])


def transform_batch(
    batch: datasets.formatting.formatting.LazyBatch,
    transforms_config: dict[str, list[Any]],
) -> dict[str, list]:
    """Transform a batch of data.

    This function applies a series of configured transformations to specified columns
    within a batch. It assumes that each transformation's `transform_all` method
    returns a list of the same length as its input.

    For 'remove_row' transforms, `np.nan` is expected in the output list for removed items.
    The 'add_row' flag's effect on overall dataset structure (like row duplication)
    is handled outside this function, based on its output.

    Args:
        batch: The input batch of data (a Hugging Face LazyBatch).
        transforms_config: A dictionary where keys are column names and values are
                           lists of transform objects to be applied to that column.

    Returns:
        A dictionary representing the transformed batch, with all original columns
        present and modified columns updated according to the transforms.
    """
    # here we should init a result directory from the batch.
    result_dict = dict(batch)
    for column_name, list_of_transforms in transforms_config.items():
        if column_name not in batch:
            logger.warning(
                f"Column '{column_name}' specified in transforms_config was not found "
                f"in the batch columns (columns: {list(batch.keys())}). Skipping transforms for this column.",
            )
            continue

        for transform_obj in list_of_transforms:
            if transform_obj.add_row:
                # here duplicate the batch
                original_values = result_dict[column_name]
                processed_values = transform_obj.transform_all(original_values)
                for key, value in result_dict.items():
                    if key != column_name:
                        result_dict[key] = value + value
                    else:
                        result_dict[key] = value + processed_values
            else:
                result_dict[column_name] = transform_obj.transform_all(result_dict[column_name])

    return result_dict


def load_dataset_from_path(data_path: str) -> datasets.DatasetDict:
    """Load dataset from various formats.

    Args:
        data_path: Path to the dataset (CSV, parquet, or HuggingFace dataset directory)

    Returns:
        A DatasetDict containing the loaded dataset
    """
    # Check if it's a directory (HuggingFace dataset)
    if os.path.isdir(data_path):
        logger.info(f"Loading dataset from directory: {data_path}")
        return datasets.load_from_disk(data_path)

    # Try to load as parquet first, then CSV
    try:
        logger.info(f"Attempting to load as parquet: {data_path}")
        dataset = datasets.load_dataset("parquet", data_files=data_path)
    except pa.ArrowInvalid:
        logger.info("Data is not in parquet format, trying CSV")
        dataset = datasets.load_dataset("csv", data_files=data_path)

    return dataset


def main(data_csv: str, config_yaml: str, out_path: str) -> None:
    """Transform the data according to the configuration.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output transformed CSV.
    """
    dataset = load_dataset_from_path(data_csv)

    dataset.set_format(type="numpy")
    # Create a DatasetProcessor object from the config and the csv
    transforms = load_transforms_from_config(config_yaml)
    logger.info("Transforms initialized successfully.")

    # Apply the transformations to the data
    dataset = dataset.map(
        transform_batch,
        batched=True,
        fn_kwargs={"transforms_config": transforms},
    )
    logger.debug(f"Dataset type: {type(dataset)}")
    dataset["train"] = dataset["train"].filter(lambda example: not any(pd.isna(value) for value in example.values()))
    dataset["train"].to_csv(out_path)
