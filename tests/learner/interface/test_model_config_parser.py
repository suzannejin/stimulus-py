"""Test the model config parser."""

import logging
import os
from typing import Any

import optuna
import pytest
import yaml

from stimulus.learner.interface import model_config_parser, model_schema

logger = logging.getLogger(__name__)


@pytest.fixture
def get_config() -> dict[str, Any]:
    """Get the config."""
    config_path = os.path.join("tests", "test_model", "conv_kan.yaml")
    with open(config_path) as f:
        return model_schema.Model(**yaml.safe_load(f))


def test_model_config_parser_suggest_parameters(get_config: model_schema.Model) -> None:
    """Test the model config parser."""
    # Create a study and trial
    study = optuna.create_study()
    trial = study.ask()
    variable_list_suggestion = model_config_parser.suggest_parameters(trial, get_config.network_params)
    logger.info(f"Variable list suggestion: {variable_list_suggestion}")

    assert variable_list_suggestion is not None
