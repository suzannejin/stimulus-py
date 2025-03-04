import inspect
import logging
from typing import Any

import optuna
import torch

from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.learner.predict import PredictWrapper

logger = logging.getLogger(__name__)

NUM_MAX_BATCHES = 500
COMPUTE_OBJECTIVE_EVERY = 10


class Objective:
    def __init__(
        self,
        model_class: torch.nn.Module,
        config: model_schema.Model,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
    ):
        self.model_class = model_class
        self.config = config
        self.train_data = train_data
        self.val_data = val_data

    def _get_optimizer(self, optimizer_name: str) -> torch.optim.Optimizer:
        """Get the optimizer from the name."""
        try:
            return getattr(torch.optim, optimizer_name)
        except AttributeError as e:
            raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim") from e

    def suggest_parameters(
        self, trial: optuna.Trial, network_params: dict[str, model_schema.TunableParameter]
    ) -> dict[str, Any]:
        """Suggest parameters for the model."""
        suggestions = {}
        for name, param in network_params.items():
            suggestion = model_config_parser.get_suggestion(name, param, trial)
            suggestions[name] = suggestion
        return suggestions

    def __call__(self, trial: optuna.Trial):
        ### suggest parameters & model setup
        model_suggestions = self.suggest_parameters(trial, self.config.network_params)
        logger.info(f"Model suggestions: {model_suggestions}")
        model_instance = self.model_class(**model_suggestions)

        ### suggest optimizer parameters
        optimizer_suggestions = self.suggest_parameters(trial, self.config.optimizer_params)
        logger.info(f"Optimizer suggestions: {optimizer_suggestions}")

        ### optimizer setup
        optimizer = self._get_optimizer(optimizer_suggestions["method"])
        optimizer_kwargs = {}
        optimizer_signature = inspect.signature(optimizer)
        for name, value in optimizer_suggestions.items():
            if name in optimizer_signature.parameters:
                optimizer_kwargs[name] = value
            else:
                logger.warning(f"Parameter {name} not found in optimizer signature, skipping")

        optimizer = optimizer(model_instance.parameters(), **optimizer_kwargs)

        ### suggest batch parameters
        batch_size = model_config_parser.get_suggestion("batch_size", self.config.data_params.batch_size, trial)
        logger.info(f"Batch size: {batch_size}")

        ### loader setup
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

        ### loss setup
        loss_dict = self.suggest_parameters(trial, self.config.loss_params)
        for key, loss_fn in loss_dict.items():
            loss_dict[key] = getattr(torch.nn, loss_fn)()
        logger.info(f"Loss parameters: {loss_dict}")

        ### perform the training loop
        batch_idx = 0
        while batch_idx < NUM_MAX_BATCHES:
            for x, y, _meta in train_loader:
                ### perform a batch update
                model_instance.batch(x=x, y=y, optimizer=optimizer, **loss_dict)
                ### update batch index
                batch_idx += 1

                ### compute objective every COMPUTE_OBJECTIVE_EVERY batches
                if batch_idx % COMPUTE_OBJECTIVE_EVERY == 0:
                    metric_dict = self.objective(model_instance, train_loader, val_loader, loss_dict)
                    logger.info(f"Objective: {metric_dict} at batch {batch_idx}")

                    # report only the validation loss (for now) #TODO use specific metric
                    trial.report(metric_dict["val_loss"], batch_idx)
                    for metric_name, metric_value in metric_dict.items():
                        trial.set_user_attr(metric_name, metric_value)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

        return metric_dict["val_loss"]  # TODO use specific metric

    def save(self, path: str):
        pass

    def objective(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_dict: dict[str, torch.nn.Module],
    ) -> dict[str, float]:
        """Compute the objective metric(s) for the tuning process."""
        metrics = [
            "loss",
            "rocauc",
            "prauc",
            "mcc",
            "f1score",
            "precision",
            "recall",
            "spearmanr",
        ]  # TODO maybe we report only a subset of metrics, given certain criteria (eg. if classification or regression)
        predict_val = PredictWrapper(
            model,
            val_loader,
            loss_dict=loss_dict,
        )
        predict_train = PredictWrapper(
            model,
            train_loader,
            loss_dict=loss_dict,
        )
        return {
            **{"val_" + metric: value for metric, value in predict_val.compute_metrics(metrics).items()},
            **{"train_" + metric: value for metric, value in predict_train.compute_metrics(metrics).items()},
        }


def tune_loop(
    objective: Objective,
    pruner: optuna.pruners.BasePruner,
    sampler: optuna.samplers.BaseSampler,
    n_trials: int,
    direction: str = "minimize",
):
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    return study
