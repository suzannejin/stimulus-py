#!/usr/bin/env python3
"""CLI module for model prediction on datasets."""

import json

import safetensors
import torch
import yaml

from stimulus.data import data_handlers
from stimulus.data.interface import data_config_parser
from stimulus.utils.model_file_interface import import_class_from_file


def load_model(model_path: str, model_config_path: str, weight_path: str) -> torch.nn.Module:
    """Dynamically loads the model from a .py file."""
    with open(model_config_path) as f:
        best_config = json.load(f)

    # Check that the model can be loaded
    model = import_class_from_file(model_path)
    model_instance = model(**best_config)

    weights = safetensors.torch.load_file(weight_path)
    model_instance.load_state_dict(weights)
    return model_instance


def load_data_config_from_path(data_path: str, data_config_path: str) -> torch.utils.data.Dataset:
    """Load the data config from a path.

    Args:
        data_path: Path to the input data file.
        data_config_path: Path to the data config file.

    Returns:
        A TorchDataset with the configured data.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.SplitTransformDict(**data_config_dict)

    encoders, input_columns, label_columns, meta_columns = data_config_parser.parse_split_transform_config(
        data_config_obj,
    )

    return data_handlers.TorchDataset(
        loader=data_handlers.DatasetLoader(
            encoders=encoders,
            input_columns=input_columns,
            label_columns=label_columns,
            meta_columns=meta_columns,
            csv_path=data_path,
        ),
    )


def predict(
    data_path: str,
    data_config_path: str,
    model_path: str,
    model_config_path: str,
    weight_path: str,
    output: str,
) -> None:
    """Run model prediction pipeline.

    Args:
        model_path: Path to the model file.
        weight_path: Path to the model weights file.
        data_path: Path to the input data file.
        output: Path to save the prediction results.
    """
    # Get the best model with best architecture and weights
    model = load_model(model_path, model_config_path, weight_path)
    dataset = load_data_config_from_path(data_path, data_config_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    # create empty tensor for predictions
    is_first_batch = True
    for x, _y, _meta in loader:
        if is_first_batch:
            predictions = model(**x)
            is_first_batch = False
        temp_predictions = model(**x)
        predictions = torch.cat((predictions, temp_predictions), dim=0)

    to_return: dict = {"predictions": predictions}
    # Predict the data
    safetensors.torch.save_file(to_return, output)
