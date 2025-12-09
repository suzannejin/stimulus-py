#!/usr/bin/env python3
"""CLI module for splitting data files.

Module currently under modification to be integrated with huggingface datasets.
Current design choices :
- Only focus on train/test splits rather than train/val/test
- Splitter class gets a dict as input
- We use save_to_disk to save the dataset to the disk with both splits at once.
"""

import logging

from stimulus.api import split
from stimulus.data.interface.dataset_interface import HuggingFaceDataset, StimulusDataset
from stimulus.data.pipelines import split as split_pipeline

logger = logging.getLogger(__name__)


def split_csv(
    data_csv: str,
    config_yaml: str,
    out_path: str,
    dataset_cls: type[StimulusDataset] = HuggingFaceDataset,
) -> None:
    """Split the data according to the configuration.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output split CSV.
        dataset_cls: The dataset class to use for loading.
    """
    # create a splitter object from the config
    splitter, split_columns = split_pipeline.load_splitters_from_config_from_path(config_yaml)

    # Load dataset using the unified loader
    dataset = dataset_cls.load_from_disk(data_csv)

    # Perform the split using the API
    # This will raise ValueError if 'test' split already exists
    split_dataset = split(dataset, splitter, split_columns)

    # Save the result
    split_dataset.save(out_path)
