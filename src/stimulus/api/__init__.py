"""Stimulus Python API module.

This module provides Python functions that wrap CLI functionality for direct use in Python scripts.
All functions work with in-memory objects (HuggingFace datasets, PyTorch models, configuration dictionaries)
instead of requiring file I/O operations.

## Usage Examples:

### Basic Data Processing
```python
import stimulus
from stimulus.api import (
    create_encoders_from_config,
    create_splitter_from_config,
)

# Load your data
dataset = datasets.load_dataset("csv", data_files="data.csv")

# Create encoders from config dict
encoder_config = {
    "columns": [
        {
            "column_name": "category",
            "column_type": "input",
            "encoder": [{"name": "LabelEncoder", "params": {"dtype": "int64"}}],
        }
    ]
}
encoders = create_encoders_from_config(encoder_config)

# Encode the dataset
encoded_dataset = stimulus.encode(dataset, encoders)

# Split the dataset
splitter_config = {
    "split": {
        "split_method": "RandomSplitter",
        "params": {"test_ratio": 0.2, "random_state": 42},
        "split_input_columns": ["category"],
    }
}
splitter, split_columns = create_splitter_from_config(splitter_config)
split_dataset = stimulus.split(encoded_dataset, splitter, split_columns)
```

### Model Training and Prediction
```python
# Define your model class
class MyModel(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.layer = torch.nn.Linear(10, hidden_size)
        # ... rest of model definition

    def batch(self, batch, optimizer=None, **loss_dict):
        # ... implement forward pass and training logic
        return loss, metrics


# Create model config
model_config = model_schema.Model(
    model_params={
        "hidden_size": model_schema.TunableParameter(
            mode="int", params={"low": 32, "high": 128}
        )
    },
    optimizer={
        "method": model_schema.TunableParameter(
            mode="categorical", params={"choices": ["Adam", "SGD"]}
        )
    },
    # ... other config
)

# Tune hyperparameters
best_config, best_model, metrics = stimulus.tune(
    dataset=split_dataset,
    model_class=MyModel,
    model_config=model_config,
    n_trials=20,
)

# Make predictions
predictions = stimulus.predict(split_dataset, best_model)
```
"""

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
