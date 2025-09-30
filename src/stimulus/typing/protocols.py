"""Protocol definitions for stimulus models."""

from typing import Any, Protocol

import torch
from torch import Tensor


class StimulusModel(Protocol):
    """Protocol for stimulus models with batch and inference methods."""

    def batch(
        self,
        batch: dict[str, Tensor],
        optimizer: Any | None = None,
        **loss_dict: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Process a batch and return loss and metrics.

        Args:
            batch: Dictionary of input tensors
            optimizer: Optional optimizer for training
            **loss_dict: Additional loss function arguments

        Returns:
            Tuple of (loss tensor, metrics dictionary)
        """
        ...

    def train_batch(
        self,
        batch: dict[str, Tensor],
        optimizer: Any,
        **loss_dict: Any,
    ) -> tuple[Tensor, dict[str, Any]] | tuple[Tensor, dict[str, Any], dict[str, Any]]:
        """Process a training batch and return loss, metrics, and optionally per-sample data.

        Args:
            batch: Dictionary of input tensors
            optimizer: Optimizer for training
            **loss_dict: Additional loss function arguments

        Returns:
            Tuple of (loss tensor, metrics dictionary) or
            Tuple of (loss tensor, metrics dictionary, per-sample dictionary)
        """
        ...

    def inference(
        self,
        batch: dict[str, Tensor],
        **loss_dict: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Run inference on a batch and return loss and metrics.

        Args:
            batch: Dictionary of input tensors
            **loss_dict: Additional loss function arguments

        Returns:
            Tuple of (loss tensor, metrics dictionary)
        """
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
