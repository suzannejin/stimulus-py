"""Module for dataset loading utilities."""

import logging
import os

import datasets
import pyarrow as pa

from stimulus.data.interface.dataset_interface import HuggingFaceDataset, StimulusDataset

logger = logging.getLogger(__name__)


def load_dataset_from_path(data_path: str) -> StimulusDataset:
    """Load dataset from various formats.

    Args:
        data_path: Path to the dataset (CSV, parquet, or HuggingFace dataset directory)

    Returns:
        A StimulusDataset containing the loaded dataset
    """
    # Check if it's a directory (HuggingFace dataset)
    if os.path.isdir(data_path):
        logger.info(f"Loading dataset from directory: {data_path}")
        dataset = datasets.load_from_disk(data_path)
    else:
        # Try to load as parquet first, then CSV
        try:
            logger.info(f"Attempting to load as parquet: {data_path}")
            dataset = datasets.load_dataset("parquet", data_files=data_path)
        except pa.ArrowInvalid:
            logger.info("Data is not in parquet format, trying CSV")
            dataset = datasets.load_dataset("csv", data_files=data_path)

    return HuggingFaceDataset(dataset)
