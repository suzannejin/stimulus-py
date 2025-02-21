#!/usr/bin/env python3
"""CLI module for checking model configuration and running initial tests."""

import logging

import click
import ray
import yaml
from torch.utils.data import DataLoader

from stimulus.data import handlertorch, loaders
from stimulus.learner import raytune_learner
from stimulus.utils import model_file_interface, yaml_data, yaml_model_schema

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to input csv file.",
)
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model file.",
)
@click.option(
    "-e",
    "--data-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to data config file.",
)
@click.option(
    "-c",
    "--model-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to yaml config training file.",
)
@click.option(
    "-w",
    "--initial-weights",
    type=click.Path(exists=True),
    help="The path to the initial weights (optional).",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    default=3,
    help="Number of samples for tuning. Overwrites tune.tune_params.num_samples in config.",
)
@click.option(
    "--ray-results-dirpath",
    type=click.Path(),
    help="Location where ray_results output dir should be written. If None, uses ~/ray_results.",
)
@click.option(
    "--debug-mode",
    is_flag=True,
    help="Activate debug mode for tuning. Default false, no debug.",
)
def check_model(
    data: str,
    model: str,
    data_config: str,
    model_config: str,
    initial_weights: str | None = None,  # noqa: ARG001
    num_samples: int = 3,
    ray_results_dirpath: str | None = None,
    debug_mode: bool = False,
) -> None:
    """Run the main model checking pipeline.

    Args:
        data: Path to input data file.
        model: Path to model file.
        data_config: Path to data config file.
        model_config: Path to model config file.
        initial_weights: Optional path to initial weights.
        num_samples: Number of samples for tuning.
        ray_results_dirpath: Directory for ray results.
        debug_mode: Whether to run in debug mode.
    """
    with open(data_config) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = yaml_data.YamlSplitTransformDict(**data_config_dict)

    with open(model_config) as file:
        model_config_dict = yaml.safe_load(file)
        model_config_obj = yaml_model_schema.Model(**model_config_dict)

    encoder_loader = loaders.EncoderLoader()
    encoder_loader.initialize_column_encoders_from_config(
        column_config=data_config_obj.columns,
    )

    logger.info("Dataset loaded successfully.")

    model_class = model_file_interface.import_class_from_file(model)

    logger.info("Model class loaded successfully.")

    ray_config_loader = yaml_model_schema.YamlRayConfigLoader(model=model_config_obj)
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

    torch_dataset = handlertorch.TorchDataset(
        data_config=data_config_obj,
        csv_path=data,
        encoder_loader=encoder_loader,
    )

    torch_dataloader = DataLoader(torch_dataset, batch_size=10, shuffle=True)

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
    model_config_obj.tune.tune_params.num_samples = num_samples

    tuner = raytune_learner.TuneWrapper(
        model_config=ray_config_model,
        data_config=data_config_obj,
        model_class=model_class,
        data_path=data,
        encoder_loader=encoder_loader,
        seed=42,
        ray_results_dir=ray_results_dirpath,
        debug=debug_mode,
    )

    logger.info("Tuner initialized successfully.")

    tuner.tune()

    logger.info("Tuning completed successfully.")
    logger.info("Checks complete")


def run() -> None:
    """Run the model checking script."""
    ray.init(address="auto", ignore_reinit_error=True)
    check_model()


if __name__ == "__main__":
    run()
