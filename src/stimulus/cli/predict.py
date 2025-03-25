#!/usr/bin/env python3
"""CLI module for model prediction on datasets."""

import torch
import importlib.util
from safetensors.torch import load_file
import inspect
from stimulus.utils.model_file_interface import import_class_from_file


def load_model(model_path, weight_path):
    """Dynamically loads the model from a .py file."""
    model = import_class_from_file(model_path)()
    # Check which is the model i loaded 
    print(model)
    weights = load_file(weight_path)
    print(weights)
    model.load_state_dict(weights)
    return model


def predict(
    data_path: str,
    model_path: str,
    weight_path: str,
    output: str
) -> None:
    """Run model prediction pipeline.

    Args:
        model_path: Path to the model file.
        weight_path: Path to the model weights file.
        data_path: Path to the input data file.
        output: Path to save the prediction results.
    """
    
    # Initialize your model (must match the architecture used in training)
    model = load_model(model_path, weight_path)

    print("Model loaded successfully.")

    
    




