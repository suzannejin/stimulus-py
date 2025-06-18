"""The test suite for the typing module.

As the typing module only contains types, the tests only check imports.
"""
# ruff: noqa: F401

import pytest

# Note: test_data_handlers_types removed as data_handlers.py was deleted


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


# Note: test_yaml_model_schema_types removed as yaml_model_schema.py was deleted
# The codebase now uses Optuna exclusively for hyperparameter tuning


@pytest.mark.skip(reason="Skipping typing tests")
def test_type_aliases() -> None:
    """Test the type aliases."""
    try:
        from stimulus.typing import Data
    except ImportError:
        pytest.fail("Failed to import Type Aliases")
