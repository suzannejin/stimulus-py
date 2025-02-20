"""Test the RayTuneLearner class."""

import os
import warnings

import pytest
import ray
import yaml
import torch

from stimulus.data.data_handlers import TorchDataset, DatasetLoader
from stimulus.data.encoding import encoders as encoders_module
from stimulus.learner.raytune_learner import TuneWrapper
from stimulus.utils.yaml_model_schema import Model, RayTuneModel, RayConfigLoader
from tests.test_model import titanic_model


@pytest.fixture
def ray_config_loader() -> RayTuneModel:
    """Load the RayTuneModel configuration."""
    with open("tests/test_model/titanic_model_cpu.yaml") as file:
        model_config = yaml.safe_load(file)
    return RayConfigLoader(Model(**model_config)).get_config()

@pytest.fixture
def get_encoders() -> dict[str, encoders_module.AbstractEncoder]:
    """Load the EncoderLoader configuration."""
    encoders = {
        "passenger_id": encoders_module.NumericEncoder(dtype=torch.int64),
        "survived": encoders_module.NumericEncoder(dtype=torch.int64),
        "pclass": encoders_module.NumericEncoder(dtype=torch.int64),
        "sex": encoders_module.StrClassificationEncoder(),
        "age": encoders_module.NumericEncoder(dtype=torch.float32),
        "sibsp": encoders_module.NumericEncoder(dtype=torch.int64),
        "parch": encoders_module.NumericEncoder(dtype=torch.int64),
        "fare": encoders_module.NumericEncoder(dtype=torch.float32),
        "embarked": encoders_module.StrClassificationEncoder(),
    }
    return encoders

@pytest.fixture
def get_input_columns() -> list[str]:
    """Get the input columns."""    
    return ["embarked", "pclass", "sex", "age", "sibsp", "parch", "fare"]

@pytest.fixture
def get_label_columns() -> list[str]:
    """Get the label columns."""
    return ["survived"]

@pytest.fixture
def get_meta_columns() -> list[str]:
    """Get the meta columns."""
    return ["passenger_id"]

@pytest.fixture
def titanic_dataset() -> TorchDataset:
    """Create a TorchDataset instance for testing."""
    return "tests/test_data/titanic/titanic_stimulus_split.csv"

@pytest.fixture
def get_train_loader(titanic_dataset: str, get_encoders: dict[str, encoders_module.AbstractEncoder], get_input_columns: list[str], get_label_columns: list[str], get_meta_columns: list[str]) -> DatasetLoader:
    """Get the DatasetLoader."""
    return DatasetLoader(csv_path=titanic_dataset,
                        encoders=get_encoders, 
                        input_columns=get_input_columns, 
                        label_columns=get_label_columns, 
                        meta_columns=get_meta_columns, 
                        split=0)

@pytest.fixture
def get_validation_loader(titanic_dataset: str, get_encoders: dict[str, encoders_module.AbstractEncoder], get_input_columns: list[str], get_label_columns: list[str], get_meta_columns: list[str]) -> DatasetLoader:
    """Get the DatasetLoader."""
    return DatasetLoader(csv_path=titanic_dataset,
                        encoders=get_encoders, 
                        input_columns=get_input_columns, 
                        label_columns=get_label_columns, 
                        meta_columns=get_meta_columns, 
                        split=1)


def test_tunewrapper_init(
    ray_config_loader: RayTuneModel,
    get_train_loader: DatasetLoader,
    get_validation_loader: DatasetLoader,
) -> None:
    """Test the initialization of the TuneWrapper class."""
    # Filter ResourceWarning during Ray shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Initialize Ray with minimal resources for testing
    ray.init(ignore_reinit_error=True)

    try:

        tune_wrapper = TuneWrapper(
            model_config=ray_config_loader,
            model_class=titanic_model.ModelTitanic,
            train_loader=get_train_loader,
            validation_loader=get_validation_loader,
            seed=42,
            ray_results_dir=os.path.abspath("tests/test_data/titanic/ray_results"),
            tune_run_name="test_run",
            debug=False,
            autoscaler=False,
        )

        assert isinstance(tune_wrapper, TuneWrapper)
    finally:
        # Force cleanup of Ray resources
        ray.shutdown()
        # Clear any temporary files
        if os.path.exists("tests/test_data/titanic/ray_results"):
            import shutil

            shutil.rmtree("tests/test_data/titanic/ray_results", ignore_errors=True)


def test_tune_wrapper_tune(
    ray_config_loader: RayTuneModel,
    get_train_loader: DatasetLoader,
    get_validation_loader: DatasetLoader,
) -> None:
    """Test the tune method of TuneWrapper class."""
    # Filter ResourceWarning during Ray shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Initialize Ray with minimal resources for testing
    ray.init(ignore_reinit_error=True)

    try:

        tune_wrapper = TuneWrapper(
            model_config=ray_config_loader,
            model_class=titanic_model.ModelTitanic,
            train_loader=get_train_loader,
            validation_loader=get_validation_loader,
            seed=42,
            ray_results_dir=os.path.abspath("tests/test_data/titanic/ray_results"),
            tune_run_name="test_run",
            debug=False,
            autoscaler=False,
        )

        tune_wrapper.tune()

    finally:
        # Force cleanup of Ray resources
        ray.shutdown()
        # Clear any temporary files
        if os.path.exists("tests/test_data/titanic/ray_results"):
            import shutil

            shutil.rmtree("tests/test_data/titanic/ray_results", ignore_errors=True)
