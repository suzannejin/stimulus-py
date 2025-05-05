"""CLI module for comparing tensors."""

import logging
from collections import defaultdict

import polars as pl
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Compute the cosine similarity between two tensors.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.

    Returns:
        The cosine similarity between the two tensors.
    """
    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()
    return torch.nn.functional.cosine_similarity(flat_tensor1, flat_tensor2, dim=0)


def discrete_compare(tensor1: torch.Tensor, tensor2: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Compute the discrete comparison between two tensors.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.

    Returns:
        The discrete comparison between the two tensors.
    """
    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()
    # for each element in the flat tensors, set to 1 if the element is greater than the threshold, otherwise set to 0
    flat_tensor1 = (flat_tensor1 > threshold).float()
    flat_tensor2 = (flat_tensor2 > threshold).float()
    # compute the dot product of the two tensors and divide by the number of elements in the tensors
    # Calculate element-wise equality (1 where both are same, 0 where different)
    # For binary tensors (0s and 1s), we want:
    # - 1 when both tensors have same value (both 0 or both 1)
    # - 0 when tensors have different values
    equality = 1.0 - torch.abs(flat_tensor1 - flat_tensor2)
    # Return the proportion of matching elements
    return torch.sum(equality) / flat_tensor1.numel()


def compare_tensors(
    tensor1: dict[str, torch.Tensor],
    tensor2: dict[str, torch.Tensor],
    mode: str = "cosine_similarity",
) -> dict[str, torch.Tensor]:
    """Compare tensors.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        mode: Mode to use for comparison.

    Returns:
        A dictionary of statistics comparing tensors
    """
    compare_modes: dict = {
        "cosine_similarity": cosine_similarity,
        "discrete_comparison": discrete_compare,
    }

    tensor_comparison: dict[str, torch.Tensor] = {}

    if mode not in compare_modes:
        raise ValueError(f"Invalid mode: {mode}, available modes: {compare_modes.keys()}")

    predictions_comparison: torch.Tensor = compare_modes[mode](tensor1["predictions"], tensor2["predictions"])
    tensor_comparison[f"{mode}"] = predictions_comparison
    common_keys: list[str] = list(set(tensor1.keys()) & set(tensor2.keys()))
    common_keys.remove("predictions")
    for key in common_keys:
        tensor_comparison[f"tensor1_{key}"] = tensor1[key]
        tensor_comparison[f"tensor2_{key}"] = tensor2[key]

    return tensor_comparison


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
