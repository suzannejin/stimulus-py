#!/usr/bin/env python3
"""CLI module for checking model configuration and running initial tests."""

import logging
import os

import optuna
import yaml

from stimulus.data import data_handlers
from stimulus.data.interface import data_config_parser
from stimulus.learner import optuna_tune
from stimulus.learner.interface import model_schema
from stimulus.utils import model_file_interface

logger = logging.getLogger(__name__)

MAX_BATCHES = 10
COMPUTE_OBJECTIVE_EVERY_N_BATCHES = 2
N_TRIALS = 5


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
    optuna_results_dirpath: str = "./optuna_results",
) -> tuple[str, str]:
    """Run the main model checking pipeline.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        data_config_path: Path to data config file.
        model_config_path: Path to model config file.
        optuna_results_dirpath: Directory for optuna results.
    """
    train_data = load_data_config_from_path(data_path, data_config_path, split=0)
    val_data = load_data_config_from_path(data_path, data_config_path, split=1)
    logger.info("Dataset loaded successfully.")

    model_class = model_file_interface.import_class_from_file(model_path)

    logger.info("Model class loaded successfully.")

    with open(model_config_path) as file:
        model_config_content = yaml.safe_load(file)
        model_config = model_schema.Model(**model_config_content)

    logger.info("Model config loaded successfully.")

    base_path = optuna_results_dirpath
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(f"{base_path}/artifacts/", exist_ok=True)
    artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=f"{base_path}/artifacts/")
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{base_path}/optuna_journal_storage.log"),
    )
    device = optuna_tune.get_device()
    objective = optuna_tune.Objective(
        model_class=model_class,
        network_params=model_config.network_params,
        optimizer_params=model_config.optimizer_params,
        data_params=model_config.data_params,
        loss_params=model_config.loss_params,
        train_torch_dataset=train_data,
        val_torch_dataset=val_data,
        artifact_store=artifact_store,
        max_batches=model_config.max_batches,
        compute_objective_every_n_batches=model_config.compute_objective_every_n_batches,
        target_metric=model_config.objective.metric,
        device=device,
    )

    logger.info(f"Objective: {objective}")
    study = optuna_tune.tune_loop(
        objective=objective,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=2),
        sampler=optuna.samplers.TPESampler(),
        n_trials=N_TRIALS,
        direction=model_config.objective.direction,
        storage=storage,
    )
    if study is None:
        raise ValueError("Study is None")
    logger.info(f"Study: {study}")
    logger.info(f"Study best trial: {study.best_trial}")
    logger.info(f"Study direction: {study.direction}")
    logger.info(f"Study best value: {study.best_value}")
    logger.info(f"Study best params: {study.best_params}")
    logger.info(f"Study trials count: {len(study.trials)}")

    for artifact_meta in optuna.artifacts.get_all_artifact_meta(study_or_trial=study):
        logger.info(artifact_meta)
    # Download the best model
    trial = study.best_trial
    best_artifact_id = trial.user_attrs["model_id"]
    file_path = str(trial.number) + "_model.safetensors"
    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path=file_path,
        artifact_id=best_artifact_id,
    )

    logger.info(f"Best model downloaded successfully to {file_path}")

    return base_path, file_path
