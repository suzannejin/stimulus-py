#!/usr/bin/env python3
"""CLI module for running raytune tuning experiment."""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import ray
import yaml
import torch

from stimulus.data import data_handlers
from stimulus.data.interface import data_config_parser
from stimulus.learner import optuna_tune
from stimulus.utils import model_file_interface, yaml_model_schema

logger = logging.getLogger(__name__)


def _raise_empty_grid() -> None:
    """Raise an error when grid results are empty."""
    raise RuntimeError("Ray Tune returned empty results grid")


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Launch check_model.")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to input csv file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to model file.",
    )
    parser.add_argument(
        "-e",
        "--data_config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to data config file.",
    )
    parser.add_argument(
        "-c",
        "--model_config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to yaml config training file.",
    )
    parser.add_argument(
        "-w",
        "--initial_weights",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="FILE",
        help="The path to the initial weights (optional).",
    )
    parser.add_argument(
        "--ray_results_dirpath",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="DIR_PATH",
        help="Location where ray_results output dir should be written. If None, uses ~/ray_results.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        nargs="?",
        const="best_model.pt",
        default="best_model.pt",
        metavar="FILE",
        help="The output file path to write the trained model to",
    )
    parser.add_argument(
        "-bm",
        "--best_metrics",
        type=str,
        required=False,
        nargs="?",
        const="best_metrics.csv",
        default="best_metrics.csv",
        metavar="FILE",
        help="The path to write the best metrics to",
    )
    parser.add_argument(
        "-bc",
        "--best_config",
        type=str,
        required=False,
        nargs="?",
        const="best_config.yaml",
        default="best_config.yaml",
        metavar="FILE",
        help="The path to write the best config to",
    )
    parser.add_argument(
        "-bo",
        "--best_optimizer",
        type=str,
        required=False,
        nargs="?",
        const="best_optimizer.pt",
        default="best_optimizer.pt",
        metavar="FILE",
        help="The path to write the best optimizer to",
    )
    parser.add_argument(
        "--tune_run_name",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="CUSTOM_RUN_NAME",
        help=(
            "Tells ray tune what the 'experiment_name' (i.e. the given tune_run name) should be. "
            "If set, the subdirectory of ray_results is named with this value and its train dir is prefixed accordingly. "
            "Default None means that ray will generate such a name on its own."
        ),
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Activate debug mode for tuning. Default false, no debug.",
    )
    return parser.parse_args()


def load_data_config_from_path(data_path: str, data_config_path: str, split: int) -> torch.utils.data.Dataset:
    """Load the data config from a path.

    Args:
        data_path: Path to the input data file.
        data_config_path: Path to the data config file.
        split: Split index to use (0=train, 1=validation, 2=test).

    Returns:
        A TorchDataset with the configured data.
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


def tune(
    data_path: str,
    model_path: str,
    data_config_path: str,
    model_config_path: str,
    initial_weights: str | None = None,  # noqa: ARG001
    optuna_results_dirpath: str | None = None,
    best_model_path: str | None = None,
    best_optimizer_path: str | None = None,
    best_metrics_path: str | None = None,
    best_config_path: str | None = None,
    *,
    debug_mode: bool = False,
    clean_ray_results: bool = False,
) -> None:
    """Run model hyperparameter tuning.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        data_config_path: Path to data config file.
        model_config_path: Path to model config file.
        initial_weights: Optional path to initial weights.
        ray_results_dirpath: Directory for ray results.
        debug_mode: Whether to run in debug mode.
        clean_ray_results: Whether to clean the ray results directory.
        output_path: Path to write the best model to.
        best_optimizer_path: Path to write the best optimizer to.
        best_metrics_path: Path to write the best metrics to.
        best_config_path: Path to write the best config to.
    """
    # Load train and validation datasets
    train_dataset = load_data_config_from_path(data_path, data_config_path, split=0)
    validation_dataset = load_data_config_from_path(data_path, data_config_path, split=1)

    # Load model class
    model_class = model_file_interface.import_class_from_file(model_path)

    # Load model config
    with open(model_config_path) as file:
        model_config_dict: dict[str, Any] = yaml.safe_load(file)
    model_config: yaml_model_schema.Model = yaml_model_schema.Model(**model_config_dict)

    # Parse Ray configuration
    ray_config_loader = yaml_model_schema.RayConfigLoader(model=model_config)
    ray_config_model = ray_config_loader.get_config()

    # Ensure output_path is provided
    if output_path is None:
        raise ValueError("output_path must not be None")

    # Ensure ray_results_dirpath is absolute
    storage_path = None
    if ray_results_dirpath:
        storage_path = str(Path(ray_results_dirpath).resolve())
        # Ensure directory exists
        Path(ray_results_dirpath).mkdir(parents=True, exist_ok=True)

        # Don't set environment variable as it's deprecated
        # Instead, we'll pass storage_path to the TuneWrapper

    try:
        # Initialize tuner with datasets
        tuner = raytune_learner.TuneWrapper(
            model_config=ray_config_model,
            model_class=model_class,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            seed=42,
            ray_results_dir=storage_path,  # Pass the path here
            debug=debug_mode,
        )

        # Run tuning
        grid_results = tuner.tune()
        if not grid_results:
            _raise_empty_grid()

        # Initialize parser with results
        parser = raytune_parser.TuneParser(result=grid_results)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save outputs using proper Result object API
        parser.save_best_model(output=output_path)
        parser.save_best_optimizer(output=best_optimizer_path)
        parser.save_best_metrics_dataframe(output=best_metrics_path)
        parser.save_best_config(output=best_config_path)

    except RuntimeError:
        logger.exception("Tuning failed")
        raise
    except KeyError:
        logger.exception("Missing expected result key")
        raise
    finally:
        if clean_ray_results and storage_path is not None:
            shutil.rmtree(Path(storage_path).resolve(), ignore_errors=True)


def run() -> None:
    """Run the model tuning script from command line."""
    ray.init(address="auto", ignore_reinit_error=True)
    args = get_args()
    tune(
        data_path=args.data,
        model_path=args.model,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        initial_weights=args.initial_weights,
        ray_results_dirpath=args.ray_results_dirpath,
        output_path=args.output,
        best_optimizer_path=args.best_optimizer,
        best_metrics_path=args.best_metrics,
        best_config_path=args.best_config,
        debug_mode=args.debug_mode,
    )


if __name__ == "__main__":
    run()
