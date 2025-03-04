"""Test the optuna tune."""

import pytest
import torch

from src.stimulus.learner import optuna_tune
from src.stimulus.utils import model_file_interface
from src.stimulus.learner.interface import model_schema
from src.stimulus.data import data_handlers
import os 
import logging

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
    model_path = os.path.join("tests", "test_model", "titanic_model_cpu.yaml")
    model_config = model_file_interface.import_class_from_file(model_path)
    return model_config

@pytest.fixture
def get_train_val_loaders()

def test_objective_init(get_model_class, get_model_config):
    """Test the objective init."""
    objective = optuna_tune.Objective(get_model_class, get_model_config)
    assert objective is not None

