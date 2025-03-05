#!/usr/bin/env python3
"""CLI module for checking model configuration and running initial tests."""

import logging

import yaml
from torch.utils.data import DataLoader

from stimulus.data import data_handlers
from stimulus.data.interface import data_config_parser
from stimulus.learner import raytune_learner
from stimulus.utils import model_file_interface, yaml_model_schema

logger = logging.getLogger(__name__)


def load_data_config_from_path(data_path: str, data_config_path: str, split: int) -> data_handlers.TorchDataset:
    """Load the data config from a path.

    Args:
        data_config_path: Path to the data config file.

    Returns:
        A tuple of the parsed configuration.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.SplitTransformDict(**data_config_dict)

    encoders, input_columns, label_columns, meta_columns = data_config_parser.parse_split_transform_config(
        data_config_obj,
    )

    return data_handlers.TorchDataset(
        loader=data_handlers.DatasetLoader(
            encoders=encoders,
            input_columns=input_columns,
            label_columns=label_columns,
            meta_columns=meta_columns,
            csv_path=data_path,
            split=split,
        ),
    )


def check_model(
    data_path: str,
    model_path: str,
    data_config_path: str,
    model_config_path: str,
    initial_weights: str | None = None,  # noqa: ARG001
    num_samples: int = 3,
    ray_results_dirpath: str | None = None,
    *,
    debug_mode: bool = False,
) -> None:
    """Run the main model checking pipeline.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        data_config_path: Path to data config file.
        model_config_path: Path to model config file.
        initial_weights: Optional path to initial weights.
        num_samples: Number of samples for tuning.
        ray_results_dirpath: Directory for ray results.
        debug_mode: Whether to run in debug mode.
    """
    train_dataset = load_data_config_from_path(data_path, data_config_path, split=0)
    validation_dataset = load_data_config_from_path(data_path, data_config_path, split=1)
    logger.info("Dataset loaded successfully.")

    model_class = model_file_interface.import_class_from_file(model_path)

    logger.info("Model class loaded successfully.")

    with open(model_config_path) as file:
        model_config_content = yaml.safe_load(file)
        model_config = yaml_model_schema.Model(**model_config_content)

    ray_config_loader = yaml_model_schema.RayConfigLoader(model=model_config)
    ray_config_dict = ray_config_loader.get_config().model_dump()
    ray_config_model = ray_config_loader.get_config()

    logger.info("Ray config loaded successfully.")

    sampled_model_params = {
        key: domain.sample() if hasattr(domain, "sample") else domain
        for key, domain in ray_config_dict["network_params"].items()
    }

    logger.info("Sampled model params loaded successfully.")

    model_instance = model_class(**sampled_model_params)

    logger.info("Model instance loaded successfully.")

    torch_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    logger.info("Torch dataloader loaded successfully.")

    # try to run the model on a single batch
    for batch in torch_dataloader:
        input_data, labels, metadata = batch
        # Log shapes of tensors in each dictionary
        for key, tensor in input_data.items():
            logger.debug(f"Input tensor '{key}' shape: {tensor.shape}")
        for key, tensor in labels.items():
            logger.debug(f"Label tensor '{key}' shape: {tensor.shape}")
        for key, list_object in metadata.items():
            logger.debug(f"Metadata lists '{key}' length: {len(list_object)}")
        output = model_instance(**input_data)
        logger.info("model ran successfully on a single batch")
        logger.debug(f"Output shape: {output.shape}")
        break

    logger.info("Model checking single pass completed successfully.")

    # override num_samples
    model_config.tune.tune_params.num_samples = num_samples

    tuner = raytune_learner.TuneWrapper(
        model_config=ray_config_model,
        model_class=model_class,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        seed=42,
        ray_results_dir=ray_results_dirpath,
        debug=debug_mode,
    )

    logger.info("Tuner initialized successfully.")

    tuner.tune()

    logger.info("Tuning completed successfully.")
    logger.info("Checks complete")
