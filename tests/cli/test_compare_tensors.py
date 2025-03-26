"""Module for testing the compare_tensors CLI command."""

import logging

import pytest

from stimulus.cli import compare_tensors

logger = logging.getLogger(__name__)


@pytest.fixture
def tensor_paths() -> list[str]:
    """Fixture for tensor paths."""
    return [
        "tests/test_data/safetensors/titanic_1.safetensors",
        "tests/test_data/safetensors/titanic_2.safetensors",
        "tests/test_data/safetensors/titanic_1_1.safetensors",
    ]


def test_compare_tensors(tensor_paths: list[str]) -> None:
    """Test the compare_tensors CLI command."""
    result = compare_tensors.compare_tensors(tensor_paths)
    assert result is not None
