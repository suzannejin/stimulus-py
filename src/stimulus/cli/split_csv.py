#!/usr/bin/env python3
"""CLI module for splitting CSV data files.

Module currently under modification to be integrated with huggingface datasets.
Current design choices :
- Only focus on train/test splits rather than train/val/test
- Splitter class gets a dict as input
- We use save_to_disk to save the dataset to the disk with both splits at once.
"""

import logging

import datasets
import pyarrow as pa
import yaml

from stimulus.data.interface import data_config_parser
from stimulus.data.splitting import splitters

logger = logging.getLogger(__name__)


def load_splitters_from_config_from_path(
    data_config_path: str,
) -> tuple[splitters.AbstractSplitter, list[str]]:
    """Load the data config from a path.

    Args:
        data_config_path: Path to the data config file.

    Returns:
        A tuple containing the splitter instance and split input columns.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.IndividualSplitConfigDict(**data_config_dict)

    return data_config_parser.parse_individual_split_config(data_config_obj)


def split_csv(data_csv: str, config_yaml: str, out_path: str, *, force: bool = False) -> None:
    """Split the data according to the configuration.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output split CSV.
        force: Overwrite the validation field if it already exists.
    """
    # create a splitter object from the config
    splitter, split_columns = load_splitters_from_config_from_path(config_yaml)
    try:
        dataset_dict = datasets.load_dataset("parquet", data_files=data_csv)
    except pa.ArrowInvalid:
        logger.info("Data is not in parquet format, trying csv")
        dataset_dict = datasets.load_dataset("csv", data_files=data_csv)

    if "test" in dataset_dict and not force:
        logger.info("Test split already exists and force was set to False. Skipping split.")
        dataset_dict.save_to_disk(out_path)
        return

    if "test" in dataset_dict and force:
        logger.info(
            "Test split already exists and force was set to True. Merging current test split into train and recalculating splits.",
        )
        dataset_dict["train"] = datasets.concatenate_datasets([dataset_dict["train"], dataset_dict["test"]])
        del dataset_dict["test"]

    dataset_dict_with_numpy_format = dataset_dict.with_format("numpy")
    column_data_dict = {}
    for col_name in split_columns:
        try:
            column_data_dict[col_name] = dataset_dict_with_numpy_format["train"][col_name]
        except KeyError as err:
            raise ValueError(
                f"Column '{col_name}' not found in dataset with columns {dataset_dict_with_numpy_format['train'].column_names}",
            ) from err

    if not column_data_dict:
        raise ValueError(
            f"No data columns were extracted for splitting. Input specified columns are {split_columns}, dataset has columns {dataset_dict_with_numpy_format['train'].column_names}",
        )
    train_indices, test_indices = splitter.get_split_indexes(column_data_dict)

    train_dataset = dataset_dict_with_numpy_format["train"].select(train_indices)
    test_dataset = dataset_dict_with_numpy_format["train"].select(test_indices)

    train_test_dataset_dict = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})
    train_test_dataset_dict.save_to_disk(out_path)
