"""stimulus-py package.

Stimulus is a package for hyperparameter tuning and data processing for deep learning models.
It provides both CLI and Python API interfaces for common ML workflows.
"""

from __future__ import annotations

# Import API functions for direct access
from stimulus.api.api import (
    check_model,
    compare_tensors,
    create_encoders_from_config,
    create_splitter_from_config,
    create_transforms_from_config,
    encode,
    load_model_from_files,
    predict,
    split,
    transform,
    tune,
)

__all__ = [
    "check_model",
    "compare_tensors",
    "create_encoders_from_config",
    "create_splitter_from_config",
    "create_transforms_from_config",
    "encode",
    "load_model_from_files",
    "predict",
    "split",
    "transform",
    "tune",
]
