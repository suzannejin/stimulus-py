"""The test suite for the typing module.

As the typing module only contains types, the tests only check imports.
"""
# ruff: noqa: F401

import pytest


@pytest.mark.skip(reason="Skipping typing tests")
def test_data_handlers_types() -> None:
    """Test the data handlers types."""
    try:
        from stimulus.typing import (
            DatasetHandler,
            DatasetLoader,
            DatasetProcessor,
            TorchDataset,
        )
    except ImportError:
        pytest.fail("Failed to import Data Handlers types")


@pytest.mark.skip(reason="Skipping typing tests")
def test_data_config_schema_types() -> None:
    """Test the data config schema types."""
    try:
        from stimulus.typing import (
            Columns,
            ColumnsEncoder,
            ConfigDict,
            GlobalParams,
            Schema,
            Split,
            SplitConfigDict,
            SplitTransformDict,
            Transform,
            TransformColumns,
            TransformColumnsTransformation,
        )
    except ImportError:
        pytest.fail("Failed to import Data Config Schema types")


@pytest.mark.skip(reason="Skipping typing tests")
def test_yaml_model_schema_types() -> None:
    """Test the YAML model schema types."""
    try:
        from stimulus.typing import (
            CustomTunableParameter,
            Data,
            Loss,
            Model,
            RunParams,
            Scheduler,
            TunableParameter,
            Tune,
            TuneParams,
        )
    except ImportError:
        pytest.fail("Failed to import YAML Model Schema types")


@pytest.mark.skip(reason="Skipping typing tests")
def test_type_aliases() -> None:
    """Test the type aliases."""
    try:
        from stimulus.typing import Data
    except ImportError:
        pytest.fail("Failed to import Type Aliases")
