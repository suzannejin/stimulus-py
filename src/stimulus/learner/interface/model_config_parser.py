"""Parse the model config."""

import logging
from typing import Callable

import optuna

from stimulus.learner.interface import model_schema

logger = logging.getLogger(__name__)


def get_pruner(pruner_config: model_schema.Pruner) -> optuna.pruners.BasePruner:
    """Get the pruner from the config."""
    available_pruners = [attr for attr in dir(optuna.pruners) if not attr.startswith("_") and attr != "TYPE_CHECKING"]
    logger.info(f"Available pruners in Optuna: {available_pruners}")

    # Check if the pruner exists with correct case
    if not hasattr(optuna.pruners, pruner_config.name):
        # Try to find a case-insensitive match
        case_matches = [attr for attr in available_pruners if attr.lower() == pruner_config.name.lower()]
        if case_matches:
            logger.info(f"Found matching pruner with different case: {case_matches[0]}")
            pruner_config.name = case_matches[0]  # Use the correct case
        else:
            raise ValueError(
                f"Pruner '{pruner_config.name}' not available in Optuna. Available pruners: {available_pruners}",
            )

    pruner_class = getattr(optuna.pruners, pruner_config.name)
    try:
        return pruner_class(**pruner_config.params)
    except TypeError as e:
        if "argument after ** must be a mapping" in str(e) and pruner_config.params is None:
            return pruner_class()
        raise


def get_sampler(sampler_config: model_schema.Sampler) -> optuna.samplers.BaseSampler:
    """Get the sampler from the config."""
    available_samplers = [attr for attr in dir(optuna.samplers) if not attr.startswith("_") and attr != "TYPE_CHECKING"]
    logger.info(f"Available samplers in Optuna: {available_samplers}")

    if not hasattr(optuna.samplers, sampler_config.name):
        # Try to find a case-insensitive match
        case_matches = [attr for attr in available_samplers if attr.lower() == sampler_config.name.lower()]
        if case_matches:
            logger.info(f"Found matching sampler with different case: {case_matches[0]}")
            sampler_config.name = case_matches[0]  # Use the correct case
        else:
            raise ValueError(
                f"Sampler '{sampler_config.name}' not available in Optuna. Available samplers: {available_samplers}",
            )

    sampler_class = getattr(optuna.samplers, sampler_config.name)
    try:
        return sampler_class(**sampler_config.params)
    except TypeError as e:
        if "argument after ** must be a mapping" in str(e) and sampler_config.params is None:
            return sampler_class()
        raise


def get_suggestion(
    name: str,
    suggestion_method_config: model_schema.TunableParameter,
    trial: optuna.trial.Trial,
) -> optuna.trial.Trial:
    """Get the suggestion method from the config."""
    trial_methods: dict[str, Callable] = {
        "categorical": trial.suggest_categorical,
        "discrete_uniform": trial.suggest_discrete_uniform,
        "float": trial.suggest_float,
        "int": trial.suggest_int,
        "loguniform": trial.suggest_loguniform,
        "uniform": trial.suggest_uniform,
    }

    if suggestion_method_config.mode not in trial_methods:
        raise ValueError(
            f"Suggestion method '{suggestion_method_config.mode}' not available in Optuna. Available suggestion methods: {trial_methods.keys()}",
        )

    return trial_methods[suggestion_method_config.mode](name=name, **suggestion_method_config.params)
