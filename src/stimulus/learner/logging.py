"""Unified experiment logging interface supporting multiple backends.

This module provides a consistent API for logging experiments to different
backends (TensorBoard, WandB) without coupling training code to specific
logging implementations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import types

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


LoggerBackend = Literal["tensorboard", "wandb", "both"]


class ExperimentLogger:
    """Unified logging interface for machine learning experiments.

    Supports TensorBoard and WandB backends with a consistent API.
    WandB integration uses TensorBoard syncing for seamless compatibility.

    Example:
        >>> # TensorBoard only
        >>> logger = ExperimentLogger(log_dir="runs/exp1", backend="tensorboard")
        >>> logger.log_scalar("train/loss", 0.5, step=0)
        >>> logger.close()

        >>> # WandB with TensorBoard sync
        >>> logger = ExperimentLogger(
        ...     log_dir="runs/exp2", backend="wandb", wandb_project="my-project"
        ... )
        >>> logger.log_scalar("train/loss", 0.5, step=0)
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str,
        backend: LoggerBackend = "tensorboard",
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        **wandb_kwargs: Any,
    ):
        """Initialize experiment logger.

        Args:
            log_dir: Directory for log files
            backend: Logging backend to use ("tensorboard", "wandb", or "both")
            wandb_project: WandB project name (required if backend includes "wandb")
            wandb_entity: WandB entity/team name (optional)
            **wandb_kwargs: Additional arguments for wandb.init()

        Raises:
            ValueError: If backend is "wandb" or "both" but wandb_project not provided
            ImportError: If wandb backend requested but wandb not installed
        """
        self.backend = backend
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Validate WandB configuration
        if backend in ["wandb", "both"]:
            if wandb_project is None:
                raise ValueError(
                    f"wandb_project is required when backend is '{backend}'",
                )
            try:
                import wandb
            except ImportError as e:
                raise ImportError(
                    "WandB backend requested but wandb is not installed. Install with: pip install wandb",
                ) from e

        # Initialize TensorBoard writer (used by both backends)
        from torch.utils.tensorboard import SummaryWriter

        self.tb_writer = SummaryWriter(log_dir=log_dir)

        # Initialize WandB if requested
        if backend in ["wandb", "both"]:
            import wandb

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                dir=log_dir,
                sync_tensorboard=True,
                **wandb_kwargs,
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Metric name (e.g., 'Train/Loss', 'val/accuracy')
            value: Scalar value to log
            step: Training step/iteration
        """
        self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        """Log multiple scalar values at once.

        Args:
            tag_value_dict: Dictionary of {tag: value} pairs
            step: Training step/iteration
        """
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)

    def log_hparams(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float],
        run_name: str | None = None,
    ) -> None:
        """Log hyperparameters and final metrics.

        Args:
            hparams: Hyperparameter dictionary
            metrics: Final metric values
            run_name: Optional run name for organization
        """
        # Convert complex values to strings for TensorBoard compatibility
        tb_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (list, tuple, dict)):
                tb_hparams[key] = str(value)
            else:
                tb_hparams[key] = value

        if run_name:
            self.tb_writer.add_hparams(tb_hparams, metrics, run_name=run_name)
        else:
            self.tb_writer.add_hparams(tb_hparams, metrics)

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the context manager and close the logger."""
        self.close()

    def close(self) -> None:
        """Close all backends and flush pending writes."""
        if self.backend in ["tensorboard", "both"] and self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()

        if self.backend in ["wandb", "both"]:
            import wandb

            wandb.finish()

    def flush(self) -> None:
        """Flush all pending writes to ensure data is written to disk."""
        if self.backend in ["tensorboard", "both"] and self.tb_writer:
            self.tb_writer.flush()

        if self.backend in ["wandb", "both"]:
            # WandB handles flushing internally, but we can be explicit
            import wandb

            wandb.finish()


def create_trial_logger(
    root_log_dir: str,
    trial_number: int,
    backend: LoggerBackend = "tensorboard",
    **kwargs: Any,
) -> ExperimentLogger:
    """Create a logger for a specific Optuna trial.

    Args:
        root_log_dir: Root logging directory
        trial_number: Optuna trial number
        backend: Logging backend
        **kwargs: Additional arguments for ExperimentLogger

    Returns:
        ExperimentLogger instance for this trial
    """
    trial_log_dir = os.path.join(root_log_dir, f"trial-{trial_number}")
    return ExperimentLogger(log_dir=trial_log_dir, backend=backend, **kwargs)
