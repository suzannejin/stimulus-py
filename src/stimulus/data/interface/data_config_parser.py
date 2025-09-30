"""Module for parsing data configs."""

import copy
import re
from typing import Any

import numpy as np
import yaml

from stimulus.data.encoding import encoders as encoders_module
from stimulus.data.interface.data_config_schema import (
    Columns,
    ConfigDict,
    EncodingConfigDict,
    IndividualSplitConfigDict,
    IndividualTransformConfigDict,
    Split,
    SplitTransformDict,
    Transform,
)
from stimulus.data.splitting import splitters as splitters_module
from stimulus.data.transforming import transforms as transforms_module


def _sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe for use as a filename.

    Args:
        filename: The original filename string.

    Returns:
        A sanitized filename string.
    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r"[^\w\-_.]", "_", filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip("_")


def _generate_split_filename(base_name: str, split_config: Split) -> str:
    """Generate a descriptive filename for a split configuration.

    Args:
        base_name: Base name for the file.
        split_config: The split configuration.

    Returns:
        A descriptive filename for the split.
    """
    # Extract split method
    method = split_config.split_method

    # Extract split ratios and convert to percentages
    min_ratio_count = 2
    if "split" in split_config.params:
        ratios = split_config.params["split"]
        if isinstance(ratios, list) and len(ratios) >= min_ratio_count:
            # Convert to percentages and format
            percentages = [int(r * 100) for r in ratios]
            ratio_str = "-".join(map(str, percentages))
        else:
            ratio_str = "default"
    else:
        ratio_str = "default"

    filename = f"{base_name}_{method}_{ratio_str}_split.yaml"
    return _sanitize_filename(filename)


def _generate_transform_filename(base_name: str, transform_config: Transform) -> str:
    """Generate a descriptive filename for a transform configuration.

    Args:
        base_name: Base name for the file.
        transform_config: The transform configuration.

    Returns:
        A descriptive filename for the transform.
    """
    # Extract transformation name
    transform_name = transform_config.transformation_name

    # Extract key parameter values for description
    param_parts = []
    seen_params = set()

    # Look through columns and transformations to find key parameters
    for column in transform_config.columns:
        for transformation in column.transformations:
            if transformation.params:
                # Extract key parameters (prioritize common ones)
                for param_name, param_value in transformation.params.items():
                    if param_name in ["std", "rate", "alpha", "beta", "factor", "amount"]:
                        # Format numeric values nicely
                        if isinstance(param_value, (int, float)):
                            if param_value == int(param_value):
                                param_desc = f"{param_name}{int(param_value)}"
                            else:
                                param_desc = f"{param_name}{param_value}"
                        else:
                            param_desc = f"{param_name}{param_value}"

                        # Only add if we haven't seen this parameter description before
                        if param_desc not in seen_params:
                            param_parts.append(param_desc)
                            seen_params.add(param_desc)

    # Create parameter string
    param_str = "_".join(param_parts[:2]) if param_parts else "default"

    filename = f"{base_name}_{transform_name}_{param_str}_transform.yaml"
    return _sanitize_filename(filename)


def _instantiate_component(module: Any, name: str, params: dict) -> Any:
    """Generic component instantiator.

    Args:
        module: The module to instantiate the component from.
        name: The name of the component to instantiate.
        params: The parameters to instantiate the component with.

    Returns:
        The instantiated component.
    """
    try:
        return getattr(module, name)(**params or {})
    except (AttributeError, TypeError) as e:
        raise ValueError(f"Failed to create {name} from {module.__name__}") from e


def create_encoders(column_config: list[Columns]) -> dict[str, encoders_module.AbstractEncoder]:
    """Factory for creating encoders from config."""

    def get_params(params: dict) -> dict:
        """Get the params with the dtype string converted to numpy dtype."""
        try:
            params_new = copy.deepcopy(params)
            dtype_str = params["dtype"]
            params_new["dtype"] = getattr(np, dtype_str)
        except AttributeError as e:
            raise ValueError(f"Invalid dtype {dtype_str} in encoder params") from e
        return params_new

    return {
        column.column_name: _instantiate_component(
            module=encoders_module,
            name=column.encoder[0].name,
            params=get_params(column.encoder[0].params),
        )
        for column in column_config
    }


def create_transforms(transform_config: list[Transform]) -> dict[str, list[Any]]:
    """Factory for creating transforms from config.

    Args:
        transform_config: List of Transform objects from the YAML config

    Returns:
        Dictionary mapping column names to lists of instantiated transform objects
    """
    transforms = {}
    for transform in transform_config:
        for column in transform.columns:
            transforms[column.column_name] = [
                _instantiate_component(
                    module=transforms_module,
                    name=transformation.name,
                    params=transformation.params,
                )
                for transformation in column.transformations
            ]
    return transforms


def create_splitter(split_config: Split) -> splitters_module.AbstractSplitter:
    """Factory for creating splitters from config."""
    return _instantiate_component(
        module=splitters_module,
        name=split_config.split_method,
        params=split_config.params,
    )


def parse_split_transform_config(
    config: SplitTransformDict,
) -> tuple[
    dict[str, encoders_module.AbstractEncoder],
    list[str],
    list[str],
    list[str],
]:
    """Parse the configuration and return a dictionary of the parsed configuration.

    Args:
        config: The configuration to parse.

    Returns:
        A tuple of the parsed configuration.
    """
    encoders = create_encoders(config.columns)
    input_columns = [column.column_name for column in config.columns if column.column_type == "input"]
    label_columns = [column.column_name for column in config.columns if column.column_type == "label"]
    meta_columns = [column.column_name for column in config.columns if column.column_type == "meta"]

    return encoders, input_columns, label_columns, meta_columns


def parse_encoding_config(
    config: EncodingConfigDict,
) -> tuple[
    dict[str, encoders_module.AbstractEncoder],
    list[str],
    list[str],
    list[str],
]:
    """Parse encoding-only configuration.

    Args:
        config: The encoding configuration to parse.

    Returns:
        A tuple of encoders and column lists.
    """
    encoders = create_encoders(config.columns)
    input_columns = [column.column_name for column in config.columns if column.column_type == "input"]
    label_columns = [column.column_name for column in config.columns if column.column_type == "label"]
    meta_columns = [column.column_name for column in config.columns if column.column_type == "meta"]

    return encoders, input_columns, label_columns, meta_columns


def parse_individual_transform_config(config: IndividualTransformConfigDict) -> dict[str, list[Any]]:
    """Parse individual transform configuration.

    Args:
        config: The individual transform configuration to parse.

    Returns:
        Dictionary mapping column names to lists of transform objects.
    """
    return create_transforms([config.transforms])


def parse_individual_split_config(
    config: IndividualSplitConfigDict,
) -> tuple[splitters_module.AbstractSplitter, list[str]]:
    """Parse individual split configuration.

    Args:
        config: The individual split configuration to parse.

    Returns:
        A tuple containing the splitter instance and split input columns.
    """
    return create_splitter(config.split), config.split.split_input_columns


def extract_transform_parameters_at_index(
    transform: Transform,
    index: int = 0,
) -> Transform:
    """Get a transform with parameters at the specified index.

    Args:
        transform: The original transform containing parameter lists
        index: Index to extract parameters from (default 0)

    Returns:
        A new transform with single parameter values at the specified index
    """
    # Create a copy of the transform
    new_transform = Transform(**transform.model_dump())

    # Process each column and transformation
    for column in new_transform.columns:
        for transformation in column.transformations:
            if transformation.params:
                # Convert each parameter list to single value at index
                new_params = {}
                for param_name, param_value in transformation.params.items():
                    if isinstance(param_value, list):
                        new_params[param_name] = param_value[index]
                    else:
                        new_params[param_name] = param_value
                transformation.params = new_params

    return new_transform


def expand_transform_parameter_combinations(
    transform: Transform,
) -> list[Transform]:
    """Get all possible transforms by extracting parameters at each valid index.

    For a transform with parameter lists, creates multiple new transforms, each containing
    single parameter values from the corresponding indices of the parameter lists.

    Args:
        transform: The original transform containing parameter lists

    Returns:
        A list of transforms, each with single parameter values from sequential indices
    """
    # Find the length of parameter lists - we only need to check the first list we find
    # since all lists must have the same length (enforced by pydantic validator)
    max_length = 1
    for column in transform.columns:
        for transformation in column.transformations:
            if transformation.params:
                list_lengths = [len(v) for v in transformation.params.values() if isinstance(v, list) and len(v) > 1]
                if list_lengths:
                    max_length = list_lengths[0]  # All lists have same length due to validator
                    break

    # Generate a transform for each index
    transforms = []
    for i in range(max_length):
        transforms.append(extract_transform_parameters_at_index(transform, i))

    return transforms


def expand_transform_list_combinations(
    transform_list: list[Transform],
) -> list[Transform]:
    """Expands a list of transforms into all possible parameter combinations.

    Takes a list of transforms where each transform may contain parameter lists,
    and expands them into separate transforms with single parameter values.
    For example, if a transform has parameters [0.1, 0.2] and [1, 2], this will
    create two transforms: one with 0.1/1 and another with 0.2/2.

    Args:
        transform_list: A list of YamlTransform objects containing parameter lists
            that need to be expanded into individual transforms.

    Returns:
        list[YamlTransform]: A flattened list of transforms where each transform
            has single parameter values instead of parameter lists. The length of
            the returned list will be the sum of the number of parameter combinations
            for each input transform.
    """
    sub_transforms = []
    for transform in transform_list:
        sub_transforms.extend(expand_transform_parameter_combinations(transform))
    return sub_transforms


def _clean_params(data: dict) -> dict:
    """Recursive cleaner for empty parameters (replaces fix_params)."""
    if isinstance(data, dict):
        return {
            k: (
                # Handle special cases for encoder/transformations lists
                [dict(e, params=e.get("params") or {}) for e in v]
                if k in ("encoder", "transformations")
                else _clean_params(v)
            )
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_clean_params(item) for item in data]
    return data


def generate_encoding_config(config: ConfigDict) -> EncodingConfigDict:
    """Generate encoding-only configuration from a master config.

    Args:
        config: The master configuration containing all components.

    Returns:
        EncodingConfigDict containing only global_params and columns.
    """
    return EncodingConfigDict(
        global_params=config.global_params,
        columns=config.columns,
    )


def generate_individual_split_configs(config: ConfigDict) -> list[IndividualSplitConfigDict]:
    """Generate individual split configurations from a master config.

    Args:
        config: The master configuration containing multiple splits.

    Returns:
        List of IndividualSplitConfigDict, one for each split in the master config.
    """
    return [
        IndividualSplitConfigDict(
            global_params=config.global_params,
            split=split,
        )
        for split in config.split
    ]


def generate_individual_transform_configs(config: ConfigDict) -> list[IndividualTransformConfigDict]:
    """Generate individual transform configurations from a master config.

    Expands parameter lists within transforms to create separate configs for each
    parameter combination.

    Args:
        config: The master configuration containing transforms with parameter lists.

    Returns:
        List of IndividualTransformConfigDict, one for each expanded transform.
    """
    # Expand all transforms to handle parameter lists
    expanded_transforms = expand_transform_list_combinations(config.transforms)

    return [
        IndividualTransformConfigDict(
            global_params=config.global_params,
            transforms=transform,
        )
        for transform in expanded_transforms
    ]


def split_config_into_components(config: ConfigDict, output_dir: str, base_name: str = "config") -> None:
    """Split a master config into separate component configs and save them.

    Args:
        config: The master configuration to split.
        output_dir: Directory to save the component configs.
        base_name: Base name for generated files (default: "config").
    """
    # Generate encoding config
    encoding_config = generate_encoding_config(config)
    encoding_data = _clean_params(encoding_config.model_dump(exclude_none=True))

    # Generate split configs
    split_configs = generate_individual_split_configs(config)

    # Generate transform configs
    transform_configs = generate_individual_transform_configs(config)

    # Save encoding config
    encoding_filename = f"{base_name}_encode.yaml"
    _save_single_yaml(encoding_data, f"{output_dir}/{encoding_filename}")

    # Save split configs
    for split_config in split_configs:
        split_data = _clean_params(split_config.model_dump(exclude_none=True))
        split_filename = _generate_split_filename(base_name, split_config.split)
        _save_single_yaml(split_data, f"{output_dir}/{split_filename}")

    # Save transform configs
    for transform_config in transform_configs:
        transform_data = _clean_params(transform_config.model_dump(exclude_none=True))
        transform_filename = _generate_transform_filename(base_name, transform_config.transforms)
        _save_single_yaml(transform_data, f"{output_dir}/{transform_filename}")


def _save_single_yaml(data: dict, file_path: str) -> None:
    """Save a single YAML config to file with consistent formatting."""
    # Maximum length for flow style numeric lists
    max_flow_style_length = 5

    def represent_dict(dumper: yaml.SafeDumper, data: dict) -> Any:
        """Custom representer for dictionaries to ensure block style."""
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=False)

    def represent_list(dumper: yaml.SafeDumper, data: list) -> Any:
        """Custom representer for lists to control flow style based on content."""
        # Use flow style only for simple numeric lists like split ratios
        is_simple_numeric = all(isinstance(i, (int, float)) for i in data) and len(data) <= max_flow_style_length
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=is_simple_numeric)

    # Create a dumper that preserves the document structure
    class ReadableDumper(yaml.SafeDumper):
        def ignore_aliases(self, _data: Any) -> bool:
            return True  # Disable anchor/alias generation

    # Register our custom representers
    ReadableDumper.add_representer(dict, represent_dict)
    ReadableDumper.add_representer(list, represent_list)
    ReadableDumper.add_representer(type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", ""))

    with open(file_path, "w") as f:
        yaml.dump(
            data,
            f,
            Dumper=ReadableDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=80,
            explicit_start=False,
            explicit_end=False,
        )
