"""CLI module for encoding CSV data files."""

import logging
from typing import Any, Optional

import datasets
import numpy as np
import yaml

from stimulus.data.interface import data_config_parser
from stimulus.data.interface.data_loading import load_dataset_from_path

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
        data_config_obj = data_config_parser.EncodingConfigDict(**data_config_dict)

    encoders, _input_columns, _label_columns, _meta_columns = data_config_parser.parse_encoding_config(
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


def main(data_path: str, config_yaml: str, out_path: str, num_proc: Optional[int] = None) -> None:
    """Encode the data according to the configuration.

    Args:
        data_path: Path to input data (CSV, parquet, or HuggingFace dataset directory).
        config_yaml: Path to config YAML file.
        out_path: Path to output encoded dataset directory.
        num_proc: Number of processes to use for encoding.
    """
    # Load the dataset
    dataset = load_dataset_from_path(data_path)

    # Set format to numpy for processing
    dataset.set_format(type="numpy")

    # Load encoders from config
    encoders = load_encoders_from_config(config_yaml)
    logger.info("Encoders initialized successfully.")
    logger.info(f"Loaded encoders for columns: {list(encoders.keys())}")

    # Identify and remove columns that aren't in the encoder configuration
    encoder_columns = set(encoders.keys())
    columns_to_remove = set()

    for split_name, split_dataset in dataset.items():
        dataset_columns = set(split_dataset.column_names)
        split_columns_to_remove = dataset_columns - encoder_columns
        columns_to_remove.update(split_columns_to_remove)
        logger.info(f"Split '{split_name}' columns to remove: {list(split_columns_to_remove)}")

    if columns_to_remove:
        logger.info(f"Removing columns not in encoder configuration: {list(columns_to_remove)}")
        dataset = dataset.remove_columns(list(columns_to_remove))

    # Apply the encoders to the data
    dataset = dataset.map(
        encode_batch,
        batched=True,
        fn_kwargs={"encoders_config": encoders},
        num_proc=num_proc,
    )

    logger.info(f"Dataset encoded successfully. Saving to: {out_path}")

    # Save the encoded dataset to disk
    dataset.save_to_disk(out_path)

    logger.info(f"Encoded dataset saved to: {out_path}")
