"""Device utilities for PyTorch model training and inference."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def resolve_device(force_device: Optional[str] = None, config_device: Optional[str] = None) -> torch.device:
    """Resolve device based on priority: force_device > config_device > auto-detection.

    Args:
        force_device: Device specified via CLI or function parameter (highest priority).
        config_device: Device specified in model configuration (medium priority).

    Returns:
        torch.device: The resolved computation device.

    Raises:
        RuntimeError: If a forced or configured device is invalid or unavailable.
    """
    if force_device is not None:
        try:
            device = torch.device(force_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Forced device '{force_device}' is not available. Please use a valid device.",
            ) from e
        else:
            logger.info(f"Using force-specified device: {force_device}")
            return device

    if config_device is not None:
        try:
            device = torch.device(config_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Device '{config_device}' specified in model configuration is not available. Please use a valid device.",
            ) from e
        else:
            logger.info(f"Using config-specified device: {config_device}")
            return device

    return get_device()


def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU) for computation.

    Returns:
        torch.device: The selected computation device
    """
    if torch.backends.mps.is_available():
        try:
            # Try to allocate a small tensor on MPS to check if it works
            device = torch.device("mps")
            # Create a small tensor and move it to MPS as a test
            test_tensor = torch.ones((1, 1)).to(device)
            del test_tensor  # Free the memory
            logger.info("Using MPS (Metal Performance Shaders) device")
        except RuntimeError as e:
            logger.warning(f"MPS available but failed to initialize: {e}")
            logger.warning("Falling back to CPU")
            return torch.device("cpu")
        else:
            return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} with {memory:.2f} GB memory")
        return device

    logger.info("Using CPU (GPU not available)")
    return torch.device("cpu")
