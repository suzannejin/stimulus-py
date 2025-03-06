"""Test the optuna tune."""

import inspect
import logging
import os

import optuna
import pytest
import torch
import yaml

from stimulus.cli.tuning import load_data_config_from_path
from stimulus.learner import optuna_tune
from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.utils import model_file_interface

logger = logging.getLogger(__name__)


@pytest.fixture
def get_model_class():
    """Get the model class."""
    model_path = os.path.join("tests", "test_model", "titanic_model.py")
    model_class = model_file_interface.import_class_from_file(model_path)
    return model_class


@pytest.fixture
def get_model_config():
    """Get the model config."""
    model_path = os.path.join("tests", "test_model", "titanic_model.yaml")
    with open(model_path) as f:
        model_config = yaml.safe_load(f)
    model_config_obj = model_schema.Model(**model_config)
    return model_config_obj


@pytest.fixture
def get_train_val_datasets():
    """Get the train and val datasets."""
    train_data = load_data_config_from_path(
        data_path=os.path.join("tests", "test_data", "titanic", "titanic_stimulus_split.csv"),
        data_config_path=os.path.join("tests", "test_data", "titanic", "titanic_unique_transform.yaml"),
        split=0,
    )
    val_data = load_data_config_from_path(
        data_path=os.path.join("tests", "test_data", "titanic", "titanic_stimulus_split.csv"),
        data_config_path=os.path.join("tests", "test_data", "titanic", "titanic_unique_transform.yaml"),
        split=1,
    )
    return train_data, val_data


def test_parameter_suggestions(get_model_config, get_model_class, get_train_val_datasets):
    """Test parameter suggestions directly without running a full study."""
    train_data, val_data = get_train_val_datasets

    # Create a study and trial
    study = optuna.create_study()
    trial = study.ask()

    # Test network params
    network_suggestions = {}
    for name, param in get_model_config.network_params.items():
        suggestion = model_config_parser.get_suggestion(name, param, trial)
        network_suggestions[name] = suggestion

    logger.info(f"Network suggestions: {network_suggestions}")

    assert 7 <= network_suggestions["nb_neurons_intermediate_layer"] <= 15
    assert 1 <= network_suggestions["nb_intermediate_layers"] <= 5
    assert network_suggestions["nb_classes"] == 2

    model_instance = get_model_class(**network_suggestions)
    logger.info(f"Model instance: {model_instance}")
    assert model_instance is not None

    optimizer_suggestions = {}
    for name, param in get_model_config.optimizer_params.items():
        suggestion = model_config_parser.get_suggestion(name, param, trial)
        optimizer_suggestions[name] = suggestion

    logger.info(f"Optimizer suggestions: {optimizer_suggestions}")

    optimizer_class = getattr(torch.optim, optimizer_suggestions["method"])
    optimizer_signature = inspect.signature(optimizer_class)
    optimizer_kwargs = {}
    for name, value in optimizer_suggestions.items():
        if name in optimizer_signature.parameters:
            optimizer_kwargs[name] = value

    optimizer = optimizer_class(model_instance.parameters(), **optimizer_kwargs)
    assert optimizer_class is not None
    assert optimizer is not None


def test_tune_loop(get_model_class, get_model_config, get_train_val_datasets):
    """Test the tune loop."""
    train_data, val_data = get_train_val_datasets
    base_path = "./artifacts"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)
    storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend("./optuna_journal_storage.log")
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=2)
    device = optuna_tune.get_device()
    objective = optuna_tune.Objective(
        model_class=get_model_class,
        network_params=get_model_config.network_params,
        optimizer_params=get_model_config.optimizer_params,
        data_params=get_model_config.data_params,
        loss_params=get_model_config.loss_params,
        train_torch_dataset=train_data,
        val_torch_dataset=val_data,
        artifact_store=artifact_store,
        max_batches=get_model_config.max_batches,
        compute_objective_every_n_batches=get_model_config.compute_objective_every_n_batches,
        target_metric=get_model_config.objective.metric,
        device=device,
    )

    logger.info(f"Objective: {objective}")
    study = optuna_tune.tune_loop(
        objective=objective,
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(),
        n_trials=get_model_config.n_trials,
        direction=get_model_config.objective.direction,
        storage=storage,
    )
    assert study is not None
    logger.debug(f"Study: {study}")
    logger.debug(f"Study best trial: {study.best_trial}")
    logger.debug(f"Study direction: {study.direction}")
    logger.debug(f"Study best value: {study.best_value}")
    logger.debug(f"Study best params: {study.best_params}")
    logger.debug(f"Study trials count: {len(study.trials)}")
    for artifact_meta in optuna.artifacts.get_all_artifact_meta(study_or_trial=study):
        logger.debug(artifact_meta)
    # Download the best model
    trial = study.best_trial
    best_artifact_id = trial.user_attrs["model_id"]
    file_path = str(trial.number) + "_model.safetensors"
    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path=file_path,
        artifact_id=best_artifact_id,
    )
