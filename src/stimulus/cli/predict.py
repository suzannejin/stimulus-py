#!/usr/bin/env python3
"""CLI module for model prediction on datasets."""

import torch
import json
import importlib.util
from safetensors.torch import load_file
from torch.utils.data import DataLoader
import pandas as pd

import yaml
from torch.utils.data import Dataset
from stimulus.data.data_handlers import TorchDataset
from stimulus.data import data_handlers

from stimulus.learner.predict import PredictWrapper
from stimulus.data.interface import data_config_parser

import inspect
from stimulus.utils.model_file_interface import import_class_from_file


def load_model(model_path, model_config_path, weight_path):
    """Dynamically loads the model from a .py file."""

    with open(model_config_path) as f:
        best_config = json.load(f)

    # Check that the model can be loaded
    model = import_class_from_file(model_path)
    model_instance = model(**best_config)

    weights = load_file(weight_path)
    model_instance.load_state_dict(weights)
    return model_instance



def load_data_config_from_path(data_path: str, data_config_path: str, split: int) -> torch.utils.data.Dataset:
    """Load the data config from a path.

    Args:
        data_path: Path to the input data file.
        data_config_path: Path to the data config file.
        split: Split index to use (0=train, 1=validation, 2=test).

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
            split=split,
        ),
    )

def predict(
    data_path: str,
    data_config_path: str,
    model_path: str,
    model_config_path: str,
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
    
    # Get the best model with best architecture and weights
    model = load_model(model_path, model_config_path, weight_path)

    dataset = load_data_config_from_path(data_path, data_config_path, split=0)

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    for x, y, _meta in loader:
        print(x)
    #    print(y)
    #    return  
        #preds = model(batch)

    # Predict the data
    #predictions = model(data)
    print("Model loaded successfully.")

    
    




