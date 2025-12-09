"""CLI module for encoding data files."""

import logging
from typing import Optional

from stimulus.data.interface.dataset_interface import HuggingFaceDataset, StimulusDataset
from stimulus.data.pipelines import encode as encode_pipeline

logger = logging.getLogger(__name__)


def main(
    data_path: str,
    config_yaml: str,
    out_path: str,
    num_proc: Optional[int] = None,
    dataset_cls: type[StimulusDataset] = HuggingFaceDataset,
) -> None:
    """Encode the data according to the configuration.

    Args:
        data_path: Path to input data (CSV, parquet, or HuggingFace dataset directory).
        config_yaml: Path to config YAML file.
        out_path: Path to output encoded dataset directory.
        num_proc: Number of processes to use for encoding.
        dataset_cls: The dataset class to use for loading.
    """
    # Load the dataset
    dataset = dataset_cls.load_from_disk(data_path)

    # Load encoders from config
    encoders = encode_pipeline.load_encoders_from_config(config_yaml)
    logger.info("Encoders initialized successfully.")
    logger.info(f"Loaded encoders for columns: {list(encoders.keys())}")

    # Identify and remove columns that aren't in the encoder configuration
    encoder_columns = set(encoders.keys())
    columns_to_remove = set()

    # Check all splits
    for split_columns in dataset.column_names.values():
        split_columns_set = set(split_columns)
        split_columns_to_remove = split_columns_set - encoder_columns
        columns_to_remove.update(split_columns_to_remove)

    if columns_to_remove:
        logger.info(f"Removing columns not in encoder configuration: {list(columns_to_remove)}")

    # Apply the encoders to the data
    dataset = dataset.map(
        encode_pipeline.encode_batch,
        batched=True,
        fn_kwargs={"encoders_config": encoders},
        remove_columns=list(columns_to_remove) if columns_to_remove else None,
        num_proc=num_proc,
    )

    logger.info(f"Dataset encoded successfully. Saving to: {out_path}")

    # Save the encoded dataset to disk
    dataset.save(out_path)

    logger.info(f"Encoded dataset saved to: {out_path}")
