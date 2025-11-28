"""Pipeline module for encoding data."""

import logging
from typing import Any

import numpy as np
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
        data_config_obj = data_config_parser.EncodingConfigDict(**data_config_dict)

    encoders, _input_columns, _label_columns, _meta_columns = data_config_parser.parse_encoding_config(
        data_config_obj,
    )

    # Return all encoders for all column types
    return encoders


def encode_batch(
    batch: dict[str, list],
    encoders_config: dict[str, Any],
) -> dict[str, list]:
    """Encode a batch of data.

    This function applies configured encoders to specified columns within a batch.
    Each encoder's `batch_encode` method is called to transform the column data.

    Args:
        batch: The input batch of data.
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
