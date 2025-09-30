"""Test the data config parser."""

import logging

import numpy as np
import pytest
import yaml

import stimulus.data.encoding.encoders as encoders_module  # TODO: should be imported from stimulus.typing instead.
from stimulus.data.interface.data_config_parser import (
    create_encoders,
    create_splitter,
    create_transforms,
    expand_transform_parameter_combinations,
    parse_split_transform_config,
)
from stimulus.data.interface.data_config_schema import (
    Columns,
    ColumnsEncoder,
    ConfigDict,
    SplitConfigDict,
    SplitTransformDict,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def split_config_path() -> str:
    """Get path to test config file.

    Returns:
        str: Path to test config file
    """
    return "tests/test_data/titanic/titanic_unique_split.yaml"


@pytest.fixture
def full_config_path() -> str:
    """Get path to test config file.

    Returns:
        str: Path to test config file
    """
    return "tests/test_data/titanic/titanic.yaml"


@pytest.fixture
def transform_config_path() -> str:
    """Get path to test config file.

    Returns:
        str: Path to test config file
    """
    return "tests/test_data/titanic/titanic_unique_transform.yaml"


@pytest.fixture
def int_config_path() -> str:
    """Get path to test config file.

    Returns:
        str: Path to test config file
    """
    return "tests/test_data/yaml_files/int_param.yaml"


@pytest.fixture
def load_transform_config(transform_config_path: str) -> dict:
    """Load the YAML config file.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: The loaded YAML config
    """
    with open(transform_config_path) as f:
        return SplitTransformDict(**yaml.safe_load(f))


@pytest.fixture
def load_split_config(split_config_path: str) -> dict:
    """Load the YAML config file.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: The loaded YAML config
    """
    with open(split_config_path) as f:
        return SplitConfigDict(**yaml.safe_load(f))


@pytest.fixture
def load_full_config(full_config_path: str) -> dict:
    """Load the YAML config file.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: The loaded YAML config
    """
    with open(full_config_path) as f:
        return ConfigDict(**yaml.safe_load(f))


@pytest.fixture
def load_int_config(int_config_path: str) -> dict:
    """Load the YAML config file.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: The loaded YAML config
    """
    with open(int_config_path) as f:
        return ConfigDict(**yaml.safe_load(f))


def test_create_encoders(load_full_config: ConfigDict) -> None:
    """Test encoder creation from config."""
    config = load_full_config
    encoders = create_encoders(config.columns)

    # Test encoder types
    assert isinstance(encoders["sex"], encoders_module.StrClassificationEncoder)
    assert isinstance(encoders["age"], encoders_module.NumericEncoder)
    assert len(encoders) == len(config.columns)

    # Test encoder parameters
    assert encoders["passenger_id"].dtype == np.int32
    assert encoders["age"].dtype == np.int8
    assert encoders["fare"].dtype == np.float32

    # Test config is not overwritten
    col_passenger_id = 0
    col_age = 4
    assert config.columns[col_passenger_id].encoder[0].params["dtype"] == "int32"
    assert config.columns[col_age].encoder[0].params["dtype"] == "int8"


def test_create_encoders_invalid_dtype() -> None:
    """Test error handling for invalid dtype."""
    column_config = [
        Columns(
            column_name="test",
            column_type="input",
            encoder=[ColumnsEncoder(name="NumericEncoder", params={"dtype": "invalid_dtype"})],
        ),
    ]
    with pytest.raises(ValueError, match="Invalid dtype"):
        create_encoders(column_config)


def test_load_int_config(load_int_config: ConfigDict) -> None:
    """Test loading a config with integer parameters."""
    config = load_int_config

    encoders = create_encoders(config.columns)

    # Test types
    assert isinstance(config, ConfigDict)
    assert isinstance(encoders, dict)


def test_create_transforms(load_split_config: SplitTransformDict) -> None:
    """Test transform pipeline creation."""
    transforms = create_transforms(load_split_config.transforms)

    # Verify transform structure
    age_transforms = transforms["age"]
    assert len(age_transforms) == 1
    assert age_transforms[0].__class__.__name__ == "GaussianNoise"
    assert "std" in age_transforms[0].__dict__


def test_create_splitter(load_full_config: ConfigDict) -> None:
    """Test splitter creation with parameters."""
    splitter = create_splitter(load_full_config.split[0])

    assert splitter.__class__.__name__ == "RandomSplit"
    assert splitter.split == [0.7, 0.3]


def test_parse_split_transform_config(load_transform_config: SplitTransformDict) -> None:
    """Test full config parsing workflow."""
    encoders, input_columns, label_columns, meta_columns = parse_split_transform_config(load_transform_config)

    assert len(encoders) == 9
    assert len(input_columns) == 7
    assert len(label_columns) == 1
    assert len(meta_columns) == 1


def test_expand_transform_parameter_combinations(load_split_config: SplitTransformDict) -> None:
    """Test parameter expansion logic."""
    expanded = expand_transform_parameter_combinations(load_split_config.transforms[0])
    assert len(expanded) == 3  # Because we have 3 std values in the YAML

    # Verify parameter unpacking
    first_params = expanded[0].columns[0].transformations[0].params
    assert first_params["std"] == 0.1
