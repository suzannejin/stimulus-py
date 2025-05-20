#!/usr/bin/env python3
"""CLI module for transforming CSV data files."""

import logging
from typing import Any

import datasets
import pyarrow
import yaml
import numpy as np

from stimulus.data.interface import data_config_parser

logger = logging.getLogger(__name__)


def load_transforms_from_config(data_config_path: str) -> dict[str, list[Any]]:
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
    for column_name, list_of_transforms in transforms_config.items():
        if column_name not in batch:
            logger.warning(
                f"Column '{column_name}' specified in transforms_config was not found "
                f"in the batch columns (columns: {list(batch.keys())}). Skipping transforms for this column.",
            )
            continue

        for transform_obj in list_of_transforms:
            if transform_obj.add_rows == True:
                # here duplicate the batch 
                """
                import pyarrow as pa
                # Assuming lb1 and lb2 are your LazyBatch objects
                # lb1 = LazyBatch(pyarrow_table_1, formatter_1)
                # lb2 = LazyBatch(pyarrow_table_2, formatter_2)

                # 1. Access the underlying pyarrow.Table objects
                table1 = lb1.pa_table
                table2 = lb2.pa_table

                # 2. Concatenate the tables (assuming compatible schemas)
                #    pyarrow.concat_tables raises an error if schemas are not compatible by default.
                concatenated_pa_table = pa.concat_tables([table1, table2])

                # 3. Create a new LazyBatch with the concatenated table
                #    You can typically reuse the formatter from one of the original LazyBatch objects.
                #    Ensure the formatter is appropriate for the concatenated_pa_table.
                concatenated_lazy_batch = LazyBatch(concatenated_pa_table, lb1.formatter)

                # Now, concatenated_lazy_batch will lazily format data from the combined table."""
                column_data_to_process = transform_obj.transform_all(column_data_to_process)
            else:
                column_data_to_process = transform_obj.transform_all(column_data_to_process)

        batch[column_name] = column_data_to_process
    return batch


def main(data: str, config_yaml: str, out_path: str) -> None:
    """Transform the data according to the configuration.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output transformed CSV.
    """
    try:
        dataset = datasets.load_dataset("parquet", data_files=data)
    except pyarrow.ArrowInvalid:
        logger.info("Data is not in parquet format, trying csv")
        dataset = datasets.load_dataset("csv", data_files=data)

    # Create a DatasetProcessor object from the config and the csv
    transforms = load_transforms_from_config(config_yaml)
    logger.info("Transforms initialized successfully.")

    # Apply the transformations to the data
    dataset = dataset.map(
        transform_batch,
        batched=True,
        fn_kwargs={"transforms_config": transforms},
    )

    dataset = dataset.filter(lambda example: not any(np.isnan(value) for value in example.values()))
