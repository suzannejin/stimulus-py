"""Pipeline module for transforming data."""

import logging
from typing import Any

import numpy as np
import yaml

from stimulus.data.interface import data_config_parser

logger = logging.getLogger(__name__)


def load_transforms_from_config(data_config_path: str) -> dict[str, list[Any]]:
    """Load the data config from a path.

    Args:
        data_config_path: Path to the data config file.

    Returns:
        A dictionary mapping column names to lists of transform objects.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.IndividualTransformConfigDict(**data_config_dict)

    return data_config_parser.parse_individual_transform_config(data_config_obj)


def transform_batch(
    batch: dict[str, list],
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
        batch: The input batch of data.
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
                        if isinstance(value, np.ndarray):
                            result_dict[key] = np.char.add(value, value)
                        else:
                            result_dict[key] = value + value
                    elif isinstance(value, np.ndarray):
                        result_dict[key] = np.char.add(value, processed_values)
                    else:
                        result_dict[key] = value + processed_values
            else:
                result_dict[column_name] = transform_obj.transform_all(result_dict[column_name])

    return result_dict
