#!/usr/bin/env python3
"""CLI module for transforming data files."""

import logging

import pandas as pd

from stimulus.data.interface.dataset_interface import HuggingFaceDataset, StimulusDataset
from stimulus.data.pipelines import transform as transform_pipeline

logger = logging.getLogger(__name__)


def transform(
    data_path: str,
    config_yaml: str,
    out_path: str,
    dataset_cls: type[StimulusDataset] = HuggingFaceDataset,
) -> None:
    """Transform the data according to the configuration.

    Args:
        data_path: Path to input data file (Parquet) or directory.
        config_yaml: Path to config YAML file.
        out_path: Path to output transformed dataset.
        dataset_cls: The dataset class to use for loading.
    """
    dataset = dataset_cls.load_from_disk(data_path)

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
