"""Module for testing the compare_tensors CLI command."""

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING

import pytest
from safetensors.torch import load_file

from stimulus.cli import compare_tensors

if TYPE_CHECKING:
    import torch

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
    mode = "cosine_similarity"
    results: dict[str, list[float | str]] = defaultdict(list)
    tensors: dict[str, dict[str, torch.Tensor]] = {path: load_file(path) for path in tensor_paths}
    for i in range(len(tensor_paths)):
        for j in range(i + 1, len(tensor_paths)):
            results["tensor1"].append(str(os.path.basename(tensor_paths[i])))
            results["tensor2"].append(str(os.path.basename(tensor_paths[j])))
            tensor1 = tensors[tensor_paths[i]]
            tensor2 = tensors[tensor_paths[j]]
            tensor_comparison = compare_tensors.compare_tensors(tensor1, tensor2, mode)
            for key, tensor in tensor_comparison.items():
                if tensor.ndim == 0:
                    results[key].append(tensor.item())
                else:
                    results[key].append(tensor.mean().item())
    logger.info(results)
    assert results is not None
