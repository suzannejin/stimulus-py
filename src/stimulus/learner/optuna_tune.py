import optuna
import torch
from src.stimulus.learner.interface import model_schema
from src.stimulus.learner.interface import model_config_parser
from src.stimulus.utils import performance
from typing import Any

import logging
logger = logging.getLogger(__name__)

class Objective:
    def __init__(
        self,
        model_class: torch.nn.Module,
        config: model_schema.Model,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        self.model_class = model_class
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

    def __call__(self, trial: optuna.Trial):
        pass

    def _get_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
        """Get the optimizer from the name."""
        try:
            return getattr(torch.optim, optimizer_name)
        except AttributeError as e:
            raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim") from e

    def suggest_parameters(self, trial: optuna.Trial, network_params: dict[str, model_schema.TunableParameter]) -> dict[str, Any]:
        """Suggest parameters for the model."""
        
        suggestions = {}
        for name, param in network_params.items():
            suggestion = model_config_parser.get_suggestion(name, param, trial)
            suggestions[name] = suggestion
        return suggestions

    def objective(self, trial: optuna.Trial):
        ### suggest parameters
        model_suggestions = self.suggest_parameters(trial, self.config.network_params)
        logger.info(f"Model suggestions: {model_suggestions}")
        model_instance = self.model_class(**model_suggestions)

        optimizer_suggestions = self.suggest_parameters(trial, self.config.optimizer_params)
        logger.info(f"Optimizer suggestions: {optimizer_suggestions}")
        ### perform a training loop
        
        ### compute the objective value

        ### return the objective value
        pass

    def save(self, path: str):
        pass

def tune_loop(objective: Objective, n_trials: int): 
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study
