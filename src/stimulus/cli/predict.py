#!/usr/bin/env python3
"""CLI module for model prediction on datasets."""

import json
import logging

import datasets
import safetensors.torch as safetensors
import torch

from stimulus.data.interface.data_loading import load_dataset_from_path
from stimulus.utils.model_file_interface import import_class_from_file

logger = logging.getLogger(__name__)


def load_model(model_path: str, model_config_path: str, weight_path: str) -> torch.nn.Module:
    """Dynamically loads the model from a .py file."""
    with open(model_config_path) as f:
        best_config = json.load(f)

    # Check that the model can be loaded
    model = import_class_from_file(model_path)
    model_instance = model(**best_config)

    weights = safetensors.load_file(weight_path)
    model_instance.load_state_dict(weights)
    return model_instance


def update_statistics(statistics: dict, temp_statistics: dict) -> dict:
    """Update the statistics with the new statistics.

    Args:
        statistics: The statistics to update.
        temp_statistics: The new statistics to update with.

    Returns:
        The updated statistics.
    """
    for key, value in temp_statistics.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                try:
                    # Check if statistics[key] is zero-dimensional and reshape it if needed
                    if statistics[key].ndim == 0:
                        statistics[key] = torch.cat([statistics[key].reshape(1), value.unsqueeze(0)], dim=0)
                    else:
                        statistics[key] = torch.cat([statistics[key], value.unsqueeze(0)], dim=0)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Error updating statistics: {e}, shape of incoming tensors is {value.shape} and in-place tensor is {statistics[key].shape}, values of those tensors are {value} and {statistics[key]}",
                    ) from e
            else:
                statistics[key] = torch.cat([statistics[key], value], dim=0)
        elif isinstance(value, (int, float, list)):
            statistics[key] = statistics[key] + value
        else:
            raise TypeError(f"Invalid statistics type: {type(value)}")

    return statistics


def convert_dict_to_tensor(data: dict) -> dict:
    """Convert a dictionary to a tensor.

    Args:
        data: The dictionary to convert.

    Returns:
        The converted dictionary.
    """
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            data[key] = torch.tensor(value)
    return data


def predict(
    data_path: str,
    model_path: str,
    model_config_path: str,
    weight_path: str,
    output: str,
    batch_size: int = 256,
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
    dataset = load_dataset_from_path(data_path)
    dataset.set_format(type="torch")
    splits = [dataset[split_name] for split_name in dataset]
    all_splits = datasets.concatenate_datasets(splits)
    loader = torch.utils.data.DataLoader(all_splits, batch_size=batch_size, shuffle=False)

    # create empty tensor for predictions
    is_first_batch = True
    for batch in loader:
        if is_first_batch:
            _loss, statistics = model.batch(batch)
            is_first_batch = False
        _loss, temp_statistics = model.batch(batch)
        statistics = update_statistics(statistics, temp_statistics)

    to_return: dict = convert_dict_to_tensor(statistics)
    # Predict the data
    safetensors.save_file(to_return, output)
