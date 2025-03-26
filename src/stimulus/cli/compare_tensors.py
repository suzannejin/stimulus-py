"""CLI module for comparing tensors."""

import logging
import os

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


def compare_tensors(tensor_paths: list[str], mode: str = "cosine_similarity") -> tuple[dict, dict]:
    """Compare tensors.

    Args:
        tensor_paths: List of paths to the tensors to compare.
        mode: Mode to use for comparison.

    Returns:
        True if the tensors are equal, False otherwise.
    """
    compare_modes: dict = {
        "cosine_similarity": cosine_similarity,
    }

    if mode not in compare_modes:
        raise ValueError(f"Invalid mode: {mode}")

    tensors = {path: load_file(path)["predictions"] for path in tensor_paths}
    pairwise_log: dict[str, list[str | float]] = {"first_tensor": [], "second_tensor": [], "similarity": []}

    statistics_log: dict[str, list[float]] = {
        "min": [],
        "max": [],
        "mean": [],
        "median": [],
        "std": [],
    }

    # Compute pairwise comparisons (without redundancy)
    results = []
    paths = list(tensors.keys())

    # Iterate over all unique pairs (i,j) where i < j
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            logger.debug(f"Comparing {path1} and {path2}")
            logger.debug(f"Tensors: {tensors[path1].shape}, {tensors[path2].shape}")
            similarity = compare_modes[mode](tensors[path1], tensors[path2])
            logger.debug(f"Similarity between {path1} and {path2}: {similarity.shape}")
            logger.debug(f"Similarity: {similarity}")
            pairwise_log["first_tensor"].append(str(os.path.basename(path1)))
            pairwise_log["second_tensor"].append(str(os.path.basename(path2)))
            pairwise_log["similarity"].append(similarity.item())
            results.append(similarity.item())

    # Compute statistics
    logger.info(f"Results: {results}")
    if len(tensor_paths) > 2:  # noqa: PLR2004
        statistics_log["min"].append(min(results))
        statistics_log["max"].append(max(results))
        statistics_log["mean"].append(torch.tensor(results).mean().item())
        statistics_log["median"].append(torch.tensor(results).median().item())
        statistics_log["std"].append(torch.tensor(results).std().item())
    else:
        statistics_log["min"].append(results[0])
        statistics_log["max"].append(results[0])
        statistics_log["mean"].append(results[0])
        statistics_log["median"].append(results[0])
        statistics_log["std"].append(0)

    return pairwise_log, statistics_log


def compare_tensors_and_save(
    tensor_paths: list[str],
    output_pairwise_logs: str,
    output_statistics_logs: str,
    mode: str = "cosine_similarity",
) -> None:
    """Compare tensors and save the results to a CSV file.

    Args:
        tensor_paths: List of paths to the tensors to compare.
        output_pairwise_logs: Path to save the pairwise logs.
        output_statistics_logs: Path to save the statistics logs.
        mode: Mode to use for comparison.
    """
    pairwise_log, statistics_log = compare_tensors(tensor_paths, mode)
    pl.DataFrame(pairwise_log).write_csv(output_pairwise_logs)
    pl.DataFrame(statistics_log).write_csv(output_statistics_logs)
