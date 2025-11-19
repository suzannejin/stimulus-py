#!/usr/bin/env python3
"""CLI module for transforming data files."""

import logging

import pandas as pd

from stimulus.data.interface.data_loading import load_dataset_from_path
from stimulus.data.pipelines import transform as transform_pipeline

logger = logging.getLogger(__name__)


def main(data_csv: str, config_yaml: str, out_path: str) -> None:
    """Transform the data according to the configuration.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output transformed CSV.
    """
    dataset = load_dataset_from_path(data_csv)

    # Create transforms from the config
    transforms = transform_pipeline.load_transforms_from_config(config_yaml)
    logger.info("Transforms initialized successfully.")

    # Apply the transformations to the data
    dataset = dataset.map(
        transform_pipeline.transform_batch,
        batched=True,
        fn_kwargs={"transforms_config": transforms},
    )
    logger.debug(f"Dataset type: {type(dataset)}")

    # Filter out NaN values
    dataset = dataset.filter(lambda example: not any(pd.isna(value) for value in example.values()))

    dataset.save(out_path)
