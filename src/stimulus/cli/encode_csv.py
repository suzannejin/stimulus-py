"""CLI module for encoding CSV data files."""

import logging
import os
from typing import Any

import datasets
import numpy as np
import pyarrow as pa
import yaml

from stimulus.data.interface import data_config_parser

logger = logging.getLogger(__name__)


def load_encoders_from_config(data_config_path: str) -> dict[str, Any]:
    """Load the encoders from the data config.

    Args:
        data_config_path: Path to the data config file.

    Returns:
        A dictionary mapping column names to encoder instances.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.SplitTransformDict(**data_config_dict)

    encoders, _input_columns, _label_columns, _meta_columns = data_config_parser.parse_split_transform_config(
        data_config_obj,
    )

    # Return all encoders for all column types
    return encoders


def encode_batch(
    batch: datasets.formatting.formatting.LazyBatch,
    encoders_config: dict[str, Any],
) -> dict[str, list]:
    """Encode a batch of data.

    This function applies configured encoders to specified columns within a batch.
    Each encoder's `batch_encode` method is called to transform the column data.

    Args:
        batch: The input batch of data (a Hugging Face LazyBatch).
        encoders_config: A dictionary where keys are column names and values are
                        encoder objects to be applied to that column.

    Returns:
        A dictionary representing the encoded batch, with all original columns
        present and encoded columns updated according to the encoders.
    """
    result_dict = dict(batch)

    for column_name, encoder in encoders_config.items():
        if column_name not in batch:
            logger.warning(
                f"Column '{column_name}' specified in encoders_config was not found "
                f"in the batch columns (columns: {list(batch.keys())}). Skipping encoding for this column.",
            )
            continue

        # Get the column data as numpy array
        column_data = np.array(batch[column_name])

        # Apply the encoder
        try:
            encoded_data = encoder.batch_encode(column_data)
            result_dict[column_name] = encoded_data.tolist() if isinstance(encoded_data, np.ndarray) else encoded_data
        except Exception:
            logger.exception(f"Failed to encode column '{column_name}'")
            raise

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


def main(data_path: str, config_yaml: str, out_path: str) -> None:
    """Encode the data according to the configuration.

    Args:
        data_path: Path to input data (CSV, parquet, or HuggingFace dataset directory).
        config_yaml: Path to config YAML file.
        out_path: Path to output encoded dataset directory.
    """
    # Load the dataset
    dataset = load_dataset_from_path(data_path)

    # Set format to numpy for processing
    dataset.set_format(type="numpy")

    # Load encoders from config
    encoders = load_encoders_from_config(config_yaml)
    logger.info("Encoders initialized successfully.")
    logger.info(f"Loaded encoders for columns: {list(encoders.keys())}")

    # Apply the encoders to the data
    dataset = dataset.map(
        encode_batch,
        batched=True,
        fn_kwargs={"encoders_config": encoders},
    )

    logger.info(f"Dataset encoded successfully. Saving to: {out_path}")

    # Save the encoded dataset to disk
    dataset.save_to_disk(out_path)

    logger.info(f"Encoded dataset saved to: {out_path}")
