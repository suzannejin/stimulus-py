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
        "discrete_comparison": discrete_compare,
    }

    if mode not in compare_modes:
        raise ValueError(f"Invalid mode: {mode}")

    tensors = {path: load_file(path) for path in tensor_paths}
    pairwise_log: dict[str, list[str | float]] = {"first_tensor": [], "second_tensor": [], "similarity": []}

    result_log: dict[str, list[torch.Tensor]] = {
        f"{mode}": [],
    }

    first_tensor_dict: dict = tensors[tensor_paths[0]]
    for key, _value in first_tensor_dict.items():
        if key != "predictions":  # Skip predictions during initialization
            result_log[f"{key}"] = []

    statistics_log: dict[str, list[torch.Tensor | float]] = {}
    # Compute pairwise comparisons (without redundancy)
    paths = list(tensors.keys())

    # Iterate over all unique pairs (i,j) where i < j
    for i in range(len(paths)):
        for key, value in tensors[paths[i]].items():
            if key != "predictions":  # ignore predictions
                result_log[f"{key}"].append(value)  # concat users statistics of each model output
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            logger.debug(f"Comparing {path1} and {path2}")
            logger.debug(f"Tensors: {tensors[path1]['predictions'].shape}, {tensors[path2]['predictions'].shape}")
            similarity = compare_modes[mode](tensors[path1]["predictions"], tensors[path2]["predictions"])
            logger.debug(f"Similarity between {path1} and {path2}: {similarity.shape}")
            logger.debug(f"Similarity: {similarity}")
            pairwise_log["first_tensor"].append(str(os.path.basename(path1)))
            pairwise_log["second_tensor"].append(str(os.path.basename(path2)))
            pairwise_log["similarity"].append(similarity.item())
            result_log[f"{mode}"].append(similarity.item())

    # Compute statistics
    logger.info(f"Results: {result_log[f'{mode}']}")
    for key in result_log:
        if len(tensor_paths) > 2:  # noqa: PLR2004
            try:
                # First concatenate tensors if they're not already a single tensor
                if isinstance(result_log[f"{key}"][0], torch.Tensor) and len(result_log[f"{key}"]) > 1:
                    combined_tensor = torch.cat(result_log[f"{key}"])
                    statistics_log[f"{key}_min"] = [combined_tensor.min().item()]
                    statistics_log[f"{key}_max"] = [combined_tensor.max().item()]
                    statistics_log[f"{key}_mean"] = [combined_tensor.mean().item()]
                    statistics_log[f"{key}_median"] = [combined_tensor.median().item()]
                    statistics_log[f"{key}_std"] = [combined_tensor.std().item()]
                else:
                    # Handle case where it's already a single tensor or list of scalars
                    statistics_log[f"{key}_min"] = [torch.tensor(result_log[f"{key}"]).min().item()]
                    statistics_log[f"{key}_max"] = [torch.tensor(result_log[f"{key}"]).max().item()]
                    statistics_log[f"{key}_mean"] = [torch.tensor(result_log[f"{key}"]).mean().item()]
                    statistics_log[f"{key}_median"] = [torch.tensor(result_log[f"{key}"]).median().item()]
                    statistics_log[f"{key}_std"] = [torch.tensor(result_log[f"{key}"]).std().item()]
            except RuntimeError as e:
                raise RuntimeError(f"Error computing statistics for {key}: {e}, values: {result_log[f'{key}']}") from e
        # Handle the case with only one or two tensors
        elif isinstance(result_log[f"{key}"][0], torch.Tensor):
            # If it's a tensor, use tensor methods
            statistics_log[f"{key}_min"] = [
                result_log[f"{key}"][0].min().item()
                if result_log[f"{key}"][0].numel() > 1
                else result_log[f"{key}"][0].item(),
            ]
            statistics_log[f"{key}_max"] = [
                result_log[f"{key}"][0].max().item()
                if result_log[f"{key}"][0].numel() > 1
                else result_log[f"{key}"][0].item(),
            ]
            statistics_log[f"{key}_mean"] = [
                result_log[f"{key}"][0].mean().item()
                if result_log[f"{key}"][0].numel() > 1
                else result_log[f"{key}"][0].item(),
            ]
            statistics_log[f"{key}_median"] = [
                result_log[f"{key}"][0].median().item()
                if result_log[f"{key}"][0].numel() > 1
                else result_log[f"{key}"][0].item(),
            ]
            statistics_log[f"{key}_std"] = [
                result_log[f"{key}"][0].std().item() if result_log[f"{key}"][0].numel() > 1 else 0,
            ]
        else:
            # If it's a scalar
            statistics_log[f"{key}_min"] = [result_log[f"{key}"][0]]
            statistics_log[f"{key}_max"] = [result_log[f"{key}"][0]]
            statistics_log[f"{key}_mean"] = [result_log[f"{key}"][0]]
            statistics_log[f"{key}_median"] = [result_log[f"{key}"][0]]
            statistics_log[f"{key}_std"] = [0]

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
