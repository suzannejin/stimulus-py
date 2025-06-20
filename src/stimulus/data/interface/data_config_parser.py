"""Module for parsing data configs."""

import copy
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
    SplitConfigDict,
    SplitTransformDict,
    Transform,
)
from stimulus.data.splitting import splitters as splitters_module
from stimulus.data.transforming import transforms as transforms_module


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


def generate_split_configs(config: ConfigDict) -> list[SplitConfigDict]:
    """Generates all possible split configuration from a YAML config.

    Takes a YAML configuration that may contain parameter lists and splits,
    and generates all unique splits into separate data configurations.

    For example, if the config has:
    - Two transforms with parameters [0.1, 0.2], [0.3, 0.4]
    - Two splits [0.7/0.3] and [0.8/0.2]
    This will generate 2 configs, 2 for each split.
        config_1:
            transform: [[0.1, 0.2], [0.3, 0.4]]
            split: [0.7, 0.3]

        config_2:
            transform: [[0.1, 0.2], [0.3, 0.4]]
            split: [0.8, 0.2]

    Args:
        config: The source YAML configuration containing transforms with
            parameter lists and multiple splits.

    Returns:
        list[SplitConfigDict]: A list of data configurations, where each
            config has a list of parameters and one split configuration. The
            length will be the product of the number of parameter combinations
            and the number of splits.
    """
    if isinstance(config, dict) and not isinstance(config, ConfigDict):
        raise TypeError("Input must be a ConfigDict object")

    sub_splits = config.split
    sub_configs = []
    for split in sub_splits:
        sub_configs.append(
            SplitConfigDict(
                global_params=config.global_params,
                columns=config.columns,
                transforms=config.transforms,
                split=split,
            ),
        )
    return sub_configs


def generate_split_transform_configs(
    config: SplitConfigDict,
) -> list[SplitTransformDict]:
    """Generates all the transform configuration for a given split.

    Takes a YAML configuration that may contain a transform or a list of transform,
    and generates all unique transform for a split into separate data configurations.

    For example, if the config has:
    - Two transforms with parameters [0.1, 0.2], [0.3, 0.4]
    - A split [0.7, 0.3]
    This will generate 2 configs, 2 for each split.
        transform_config_1:
            transform: [0.1, 0.2]
            split: [0.7, 0.3]

        transform_config_2:
            transform: [0.3, 0.4]
            split: [0.7, 0.3]

    Args:
        config: The source YAML configuration containing each
            a split with transforms with parameters lists

    Returns:
        list[SplitTransformDict]: A list of data configurations, where each
            config has a list of parameters and one split configuration. The
            length will be the product of the number of parameter combinations
            and the number of splits.
    """
    if isinstance(config, dict) and not isinstance(
        config,
        SplitConfigDict,
    ):
        raise TypeError("Input must be a list of SplitConfigDict")

    sub_transforms = expand_transform_list_combinations(config.transforms)
    split_transform_config: list[SplitTransformDict] = []
    for transform in sub_transforms:
        split_transform_config.append(
            SplitTransformDict(
                global_params=config.global_params,
                columns=config.columns,
                transforms=transform,
                split=config.split,
            ),
        )
    return split_transform_config


def dump_yaml_list_into_files(
    yaml_list: list[SplitConfigDict],
    directory_path: str,
    base_name: str,
    len_simple_numeric: int = 5,
) -> None:
    """Dumps YAML configurations to files with consistent, readable formatting."""

    def represent_dict(dumper: yaml.SafeDumper, data: dict) -> Any:
        """Custom representer for dictionaries to ensure block style."""
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=False)

    def represent_list(dumper: yaml.SafeDumper, data: list) -> Any:
        """Custom representer for lists to control flow style based on content."""
        # Use flow style only for simple numeric lists like split ratios
        is_simple_numeric = all(isinstance(i, (int, float)) for i in data) and len(data) <= len_simple_numeric
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=is_simple_numeric)

    # Create a dumper that preserves the document structure
    class ReadableDumper(yaml.SafeDumper):
        def ignore_aliases(self, _data: Any) -> bool:
            return True  # Disable anchor/alias generation

    # Register our custom representers
    ReadableDumper.add_representer(dict, represent_dict)
    ReadableDumper.add_representer(list, represent_list)
    ReadableDumper.add_representer(type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", ""))

    for i, yaml_dict in enumerate(yaml_list):
        data = _clean_params(yaml_dict.model_dump(exclude_none=True))

        with open(f"{directory_path}/{base_name}_{i}.yaml", "w") as f:
            yaml.dump(
                data,
                f,
                Dumper=ReadableDumper,
                default_flow_style=False,  # Default to block style for readability
                sort_keys=False,
                indent=2,
                width=80,  # Set reasonable line width
                explicit_start=False,
                explicit_end=False,
            )


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


def split_config_into_components(config: ConfigDict, output_dir: str) -> None:
    """Split a master config into separate component configs and save them.

    Args:
        config: The master configuration to split.
        output_dir: Directory to save the component configs.
    """
    # Generate encoding config
    encoding_config = generate_encoding_config(config)
    encoding_data = _clean_params(encoding_config.model_dump(exclude_none=True))

    # Generate split configs
    split_configs = generate_individual_split_configs(config)

    # Generate transform configs
    transform_configs = generate_individual_transform_configs(config)

    # Save encoding config
    _save_single_yaml(encoding_data, f"{output_dir}/encode.yaml")

    # Save split configs
    for i, split_config in enumerate(split_configs, 1):
        split_data = _clean_params(split_config.model_dump(exclude_none=True))
        _save_single_yaml(split_data, f"{output_dir}/split{i}.yaml")

    # Save transform configs
    for i, transform_config in enumerate(transform_configs, 1):
        transform_data = _clean_params(transform_config.model_dump(exclude_none=True))
        _save_single_yaml(transform_data, f"{output_dir}/transform{i}.yaml")


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
