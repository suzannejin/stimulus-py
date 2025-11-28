"""Protocol definitions for stimulus models."""

from typing import Any, Protocol

import torch
from torch import Tensor


class StimulusModel(Protocol):
    """Protocol for stimulus models compatible with the Stimulus training framework.

    Models implementing this protocol work seamlessly with the Optuna hyperparameter
    tuning pipeline and support flexible experiment logging backends (TensorBoard, WandB).

    Key changes from previous interface:
    - Uses ExperimentLogger for backend-agnostic logging
    - Models define their own loss functions internally
    - Validation handled by model's validate() method
    - No more per-sample tracking requirement
    """

    def train_batch(
        self,
        batch: dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
        logger: Any,  # ExperimentLogger from stimulus.learner.logging
        global_step: int,
    ) -> tuple[float, dict[str, float]]:
        """Train on a single batch.

        Args:
            batch: Dictionary of batch data (tensors and metadata)
            optimizer: PyTorch optimizer instance
            logger: ExperimentLogger for metrics logging
            global_step: Current training step for logging

        Returns:
            Tuple of (loss value, metrics dictionary)
            - loss: Scalar loss value (float)
            - metrics: Dictionary of training metrics (e.g., {'accuracy': 0.95})

        Example:
            >>> loss, metrics = model.train_batch(
            ...     batch={"input": x, "target": y},
            ...     optimizer=optimizer,
            ...     logger=logger,
            ...     global_step=100,
            ... )
            >>> # loss = 0.234, metrics = {'accuracy': 0.89}
        """
        ...

    def validate(
        self,
        data_loader: torch.utils.data.DataLoader,
        logger: Any | None = None,  # ExperimentLogger
        global_step: int | None = None,
    ) -> dict[str, float]:
        """Validate the model on a data loader.

        Args:
            data_loader: PyTorch DataLoader with validation data
            logger: Optional ExperimentLogger for metrics logging
            global_step: Optional step for logging (if logger provided)

        Returns:
            Dictionary of validation metrics (e.g., {'loss': 0.123, 'accuracy': 0.92})

        Note:
            The model should handle all metric computation internally, including:
            - Batch-by-batch accumulation
            - Averaging across batches
            - Any domain-specific metrics

            If logger and global_step are provided, the model may log metrics
            to the experiment logger during validation.

        Example:
            >>> metrics = model.validate(data_loader=val_loader, logger=logger, global_step=100)
            >>> # metrics = {'loss': 0.156, 'accuracy': 0.91, 'f1': 0.88}
        """
        ...

    def train(self) -> None:
        """Set model to training mode."""
        ...

    def eval(self) -> None:
        """Set model to evaluation mode."""
        ...

    def to(self, device: torch.device) -> "StimulusModel":
        """Move model to device.

        Args:
            device: Target device

        Returns:
            Model on the target device
        """
        ...

    def parameters(self) -> Any:
        """Return model parameters for optimizer.

        Returns:
            Iterator over model parameters
        """
        ...
