"""Test the model schema."""

import pytest
import os
import yaml
import logging
from src.stimulus.learner.interface.model_schema import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def get_config():
    """Get the config."""
    config_path = os.path.join("tests", "test_model", "titanic_model_cpu.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def test_model_schema(get_config):
    """Test the model schema."""
    model = Model(**get_config)
    assert model.network_params["nb_neurons_intermediate_layer"].params["low"] == 7
    assert model.network_params["nb_neurons_intermediate_layer"].params["high"] == 15
    assert model.network_params["nb_neurons_intermediate_layer"].mode == "int"
    assert model.network_params["nb_intermediate_layers"].params["low"] == 1
    assert model.network_params["nb_intermediate_layers"].params["high"] == 5
    assert model.network_params["nb_intermediate_layers"].mode == "int"
    assert model.network_params["nb_classes"].params["choices"] == [2]
    assert model.network_params["nb_classes"].mode == "categorical"
    assert model.optimizer_params["method"].params["choices"] == ["Adam", "SGD"]
    assert model.optimizer_params["method"].mode == "categorical"
    assert model.optimizer_params["lr"].params["low"] == 0.0001
    assert model.optimizer_params["lr"].params["high"] == 0.1
    assert model.optimizer_params["lr"].mode == "loguniform"
    assert model.loss_params.loss_fn.params["choices"] == ["CrossEntropyLoss"]
    assert model.loss_params.loss_fn.mode == "categorical"
    assert model.data_params.batch_size.params["choices"] == [16, 32, 64, 128, 256]
    assert model.data_params.batch_size.mode == "categorical"
    assert model.pruner.params["n_warmup_steps"] == 10
    assert model.pruner.params["n_startup_trials"] == 2
    assert model.sampler.name == "TPESampler"
    assert model.sampler.params is None