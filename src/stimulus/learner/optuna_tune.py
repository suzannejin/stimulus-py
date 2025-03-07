"""Optuna tuning module."""

import inspect
import logging
import os
import uuid
from typing import Any

import optuna
import torch
from safetensors.torch import save_model as safe_save_model

from stimulus.learner.interface import model_config_parser, model_schema
from stimulus.learner.predict import PredictWrapper

logger = logging.getLogger(__name__)


class Objective:
    """Objective class for Optuna tuning."""

    def __init__(
        self,
        model_class: torch.nn.Module,
        network_params: dict[str, model_schema.TunableParameter],
        optimizer_params: dict[str, model_schema.TunableParameter],
        data_params: dict[str, model_schema.TunableParameter],
        loss_params: dict[str, model_schema.TunableParameter],
        train_torch_dataset: torch.utils.data.Dataset,
        val_torch_dataset: torch.utils.data.Dataset,
        artifact_store: Any,
        max_batches: int = 1000,
        compute_objective_every_n_batches: int = 50,
        target_metric: str = "val_loss",
        device: torch.device | None = None,
    ):
        """Initialize the Objective class.

        Args:
            model_class: The model class to be tuned.
            network_params: The network parameters to be tuned.
            optimizer_params: The optimizer parameters to be tuned.
            data_params: The data parameters to be tuned.
            loss_params: The loss parameters to be tuned.
            train_torch_dataset: The training dataset.
            val_torch_dataset: The validation dataset.
            artifact_store: The artifact store to save the model and optimizer.
            max_batches: The maximum number of batches to train.
            compute_objective_every_n_batches: The number of batches to compute the objective.
            target_metric: The target metric to optimize.
            device: The device to run the training on.
        """
        self.model_class = model_class
        self.network_params = network_params
        self.optimizer_params = optimizer_params
        self.data_params = data_params
        self.loss_params = loss_params
        self.train_torch_dataset = train_torch_dataset
        self.val_torch_dataset = val_torch_dataset
        self.artifact_store = artifact_store
        self.target_metric = target_metric
        self.max_batches = max_batches
        self.compute_objective_every_n_batches = compute_objective_every_n_batches
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def _get_optimizer(self, optimizer_name: str) -> type[torch.optim.Optimizer]:
        """Get the optimizer from the name."""
        try:
            return getattr(torch.optim, optimizer_name)
        except AttributeError as e:
            raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim") from e

    def suggest_parameters(
        self,
        trial: optuna.Trial,
        params: dict[str, model_schema.TunableParameter],
    ) -> dict[str, Any]:
        """Suggest parameters for the model."""
        suggestions = {}
        for name, param in params.items():
            suggestion = model_config_parser.get_suggestion(name, param, trial)
            suggestions[name] = suggestion
        return suggestions

    def __call__(self, trial: optuna.Trial):
        """Execute a full training trial and return the objective metric value."""
        # Setup phase
        model_instance = self._setup_model(trial)
        optimizer = self._setup_optimizer(trial, model_instance)
        train_loader, val_loader = self._setup_data_loaders(trial)
        loss_dict = self._setup_loss_functions(trial)

        # Training loop
        batch_idx: int = 0
        metric_dict: dict = {}

        while batch_idx < self.max_batches:
            for x, y, _meta in train_loader:
                try:
                    device_x = {key: value.to(self.device, non_blocking=True) for key, value in x.items()}
                    device_y = {key: value.to(self.device, non_blocking=True) for key, value in y.items()}

                    # Perform a batch update
                    model_instance.batch(x=device_x, y=device_y, optimizer=optimizer, **loss_dict)

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                        logger.warning(f"CUDA out of memory during training: {e}")
                        logger.warning("Falling back to CPU for this trial")
                        cpu_device = torch.device("cpu")
                        model_instance = model_instance.to(cpu_device)
                        # Consider adjusting batch size or other parameters
                        device_x = {key: value.to(cpu_device) for key, value in x.items()}
                        device_y = {key: value.to(cpu_device) for key, value in y.items()}
                        # Retry the batch
                        model_instance.batch(x=device_x, y=device_y, optimizer=optimizer, **loss_dict)
                    else:
                        raise

                batch_idx += 1
                # Compute objective periodically
                if batch_idx % self.compute_objective_every_n_batches == 0:
                    # Evaluate current model performance
                    metric_dict = self.objective(model_instance, train_loader, val_loader, loss_dict)
                    logger.info(f"Objective: {metric_dict} at batch {batch_idx}")

                    # Report to Optuna
                    trial.report(metric_dict[self.target_metric], batch_idx)
                    for metric_name, metric_value in metric_dict.items():
                        trial.set_user_attr(metric_name, metric_value)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        self.save_checkpoint(trial, model_instance, optimizer)
                        raise optuna.TrialPruned()  # noqa: RSE102

                if batch_idx >= self.max_batches:
                    break

        # Final checkpoint and return objective value
        self.save_checkpoint(trial, model_instance, optimizer)
        return metric_dict[self.target_metric]

    def _setup_model(self, trial: optuna.Trial) -> torch.nn.Module:
        """Setup the model for the trial."""
        model_suggestions = self.suggest_parameters(trial, self.network_params)
        logger.info(f"Model suggestions: {model_suggestions}")
        model_instance = self.model_class(**model_suggestions)

        try:
            model_instance = model_instance.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
        except RuntimeError as e:
            if self.device.type == "cuda":
                logger.warning(f"Failed to move model to GPU: {e}")
                logger.warning("Falling back to CPU")
                self.device = torch.device("cpu")
                model_instance = model_instance.to(self.device)
            else:
                raise
        return model_instance

    def _setup_optimizer(self, trial: optuna.Trial, model_instance: torch.nn.Module) -> torch.optim.Optimizer:
        """Setup the optimizer for the trial."""
        optimizer_suggestions = self.suggest_parameters(trial, self.optimizer_params)
        logger.info(f"Optimizer suggestions: {optimizer_suggestions}")

        optimizer_class = self._get_optimizer(optimizer_suggestions["method"])
        optimizer_kwargs = {}
        optimizer_signature = inspect.signature(optimizer_class).parameters
        for k, v in optimizer_suggestions.items():
            if k in optimizer_signature and k != "method":
                optimizer_kwargs[k] = v
            elif k != "method":
                logger.warning(f"Parameter '{k}' not found in optimizer signature, skipping")

        return optimizer_class(model_instance.parameters(), **optimizer_kwargs)

    def _setup_data_loaders(
        self,
        trial: optuna.Trial,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Setup the data loaders for the trial."""
        batch_size = self.suggest_parameters(trial, self.data_params)["batch_size"]
        logger.info(f"Batch size: {batch_size}")

        train_loader = torch.utils.data.DataLoader(
            self.train_torch_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_torch_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, val_loader

    def _setup_loss_functions(self, trial: optuna.Trial) -> dict[str, torch.nn.Module]:
        """Setup the loss functions for the trial."""
        loss_dict = self.suggest_parameters(trial, self.loss_params)
        for key, loss_fn in loss_dict.items():
            loss_dict[key] = getattr(torch.nn, loss_fn)()
        logger.info(f"Loss parameters: {loss_dict}")
        return loss_dict

    def save_checkpoint(
        self,
        trial: optuna.Trial,
        model_instance: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Save the model and optimizer to the trial."""
        unique_id = str(uuid.uuid4())[:8]
        model_path = f"{trial.number}_{unique_id}_model.safetensors"
        optimizer_path = f"{trial.number}_{unique_id}_optimizer.pt"
        safe_save_model(model_instance, model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        artifact_id_model = optuna.artifacts.upload_artifact(
            artifact_store=self.artifact_store,
            file_path=model_path,
            study_or_trial=trial.study,
        )
        artifact_id_optimizer = optuna.artifacts.upload_artifact(
            artifact_store=self.artifact_store,
            file_path=optimizer_path,
            study_or_trial=trial.study,
        )
        # delete the files from the local filesystem
        try:
            os.remove(model_path)
            os.remove(optimizer_path)
        except FileNotFoundError:
            logger.info(f"File was already deleted: {model_path} or {optimizer_path}, most likely due to pruning")
        trial.set_user_attr("model_id", artifact_id_model)
        trial.set_user_attr("optimizer_id", artifact_id_optimizer)

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
            device=self.device,
        )
        predict_train = PredictWrapper(
            model,
            train_loader,
            loss_dict=loss_dict,
            device=self.device,
        )
        return {
            **{"val_" + metric: value for metric, value in predict_val.compute_metrics(metrics).items()},
            **{"train_" + metric: value for metric, value in predict_train.compute_metrics(metrics).items()},
        }


def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU) for computation.

    Returns:
        torch.device: The selected computation device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} with {memory:.2f} GB memory")
        return device
    logger.info("Using CPU (GPU not available)")
    return torch.device("cpu")


def tune_loop(
    objective: Objective,
    pruner: optuna.pruners.BasePruner,
    sampler: optuna.samplers.BaseSampler,
    n_trials: int,
    direction: str,
    storage: optuna.storages.BaseStorage | None = None,
) -> optuna.Study:
    """Run the tuning loop.

    Args:
        objective: The objective function to optimize.
        pruner: The pruner to use.
        sampler: The sampler to use.
        n_trials: The number of trials to run.
        direction: The direction to optimize.
        storage: The storage to use.

    Returns:
        The study object.
    """
    if storage is None:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, storage=storage)
    study.optimize(objective, n_trials=n_trials)
    return study
