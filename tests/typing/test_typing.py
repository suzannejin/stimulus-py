"""The test suite for the typing module.

As the typing module only contains types, the tests only check imports.
"""

import pytest

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
    YamlData,
)

# Note: test_data_handlers_types removed as data_handlers.py was deleted


@pytest.mark.skip(reason="Skipping typing tests")
def test_data_config_schema_types() -> None:
    """Test the data config schema types."""
    try:
        # Test that imports work - already imported at top level
        assert Columns is not None
        assert ColumnsEncoder is not None
        assert ConfigDict is not None
        assert GlobalParams is not None
        assert Schema is not None
        assert Split is not None
        assert SplitConfigDict is not None
        assert SplitTransformDict is not None
        assert Transform is not None
        assert TransformColumns is not None
        assert TransformColumnsTransformation is not None
    except ImportError:
        pytest.fail("Failed to import Data Config Schema types")


# Note: test_yaml_model_schema_types removed as yaml_model_schema.py was deleted
# The codebase now uses Optuna exclusively for hyperparameter tuning


@pytest.mark.skip(reason="Skipping typing tests")
def test_type_aliases() -> None:
    """Test the type aliases."""
    try:
        # Test that imports work - already imported at top level
        assert YamlData is not None
    except ImportError:
        pytest.fail("Failed to import Type Aliases")
