"""CLI module for comparing tensors."""

import logging
from collections import defaultdict

import polars as pl
import torch
from safetensors.torch import load_file

from stimulus.learner.compare import compare_tensors

logger = logging.getLogger(__name__)


def compare_tensors_and_save(
    tensor_paths: list[str],
    output_logs: str,
    mode: str = "cosine_similarity",
) -> None:
    """Compare tensors and save the results to a CSV file.

    Args:
        tensor_paths: List of paths to the tensors to compare.
        output_logs: Path to save the logs.
        mode: Mode to use for comparison.
    """
    results: dict[str, list[float | str]] = defaultdict(list)
    tensors: dict[str, dict[str, torch.Tensor]] = {path: load_file(path) for path in tensor_paths}
    for i in range(len(tensor_paths)):
        for j in range(i + 1, len(tensor_paths)):
            tensor1 = tensors[tensor_paths[i]]
            tensor2 = tensors[tensor_paths[j]]
            tensor_comparison = compare_tensors(tensor1, tensor2, mode)
            for key, tensor in tensor_comparison.items():
                if tensor.ndim == 0:
                    results[key].append(tensor.item())
                else:
                    results[key].append(tensor.mean().item())
    pl.DataFrame(results).write_csv(output_logs)
