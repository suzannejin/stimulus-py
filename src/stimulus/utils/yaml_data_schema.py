"""Utility module for handling YAML configuration files and their validation."""

from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, ValidationError, field_validator


class GlobalParams(BaseModel):
    """Model for global parameters in YAML configuration."""

    seed: int


class ColumnsEncoder(BaseModel):
    """Model for column encoder configuration."""

    name: str
    params: Optional[dict[str, Union[str, list[Any]]]]  # Allow both string and list values


class Columns(BaseModel):
    """Model for column configuration."""

    column_name: str
    column_type: str
    data_type: str
    encoder: list[ColumnsEncoder]


class TransformColumnsTransformation(BaseModel):
    """Model for column transformation configuration."""

    name: str
    params: Optional[dict[str, Union[list[Any], float]]]  # Allow both list and float values


class TransformColumns(BaseModel):
    """Model for transform columns configuration."""

    column_name: str
    transformations: list[TransformColumnsTransformation]


class Transform(BaseModel):
    """Model for transform configuration."""

    transformation_name: str
    columns: list[TransformColumns]

    @field_validator("columns")
    @classmethod
    def validate_param_lists_across_columns(
        cls,
        columns: list[TransformColumns],
    ) -> list[TransformColumns]:
        """Validate that parameter lists across columns have consistent lengths.

        Args:
            columns: List of transform columns to validate

        Returns:
            The validated columns list
        """
        # Get all parameter list lengths across all columns and transformations
        all_list_lengths: set[int] = set()

        for column in columns:
            for transformation in column.transformations:
                if transformation.params and any(
                    isinstance(param_value, list) and len(param_value) > 0
                    for param_value in transformation.params.values()
                ):
                    all_list_lengths.update(
                        len(param_value)
                        for param_value in transformation.params.values()
                        if isinstance(param_value, list) and len(param_value) > 0
                    )

        # Skip validation if no lists found
        if not all_list_lengths:
            return columns

        # Check if all lists either have length 1, or all have the same length
        all_list_lengths.discard(1)  # Remove length 1 as it's always valid
        if len(all_list_lengths) > 1:  # Multiple different lengths found
            raise ValueError(
                "All parameter lists across columns must either contain one element or have the same length",
            )

        return columns


class Split(BaseModel):
    """Model for split configuration."""

    split_method: str
    params: dict[str, list[float]]  # More specific type for split parameters
    split_input_columns: list[str]


class ConfigDict(BaseModel):
    """Model for main YAML configuration."""

    global_params: GlobalParams
    columns: list[Columns]
    transforms: list[Transform]
    split: list[Split]


# TODO: Rename this class to SplitConfigDict
class SplitConfigDict(BaseModel):
    """Model for sub-configuration generated from main config."""

    global_params: GlobalParams
    columns: list[Columns]
    transforms: list[Transform]
    split: Split


class SplitTransformDict(BaseModel):
    """Model for sub-configuration generated from main config."""

    global_params: GlobalParams
    columns: list[Columns]
    transforms: Transform
    split: Split


class Schema(BaseModel):
    """Model for validating YAML schema."""

    config: ConfigDict


class SplitSchema(BaseModel):
    """Model for validating a Split YAML schema."""

    config: ConfigDict


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
        yaml_config: The source YAML configuration containing transforms with
            parameter lists and multiple splits.

    Returns:
        list[YamlSubConfigDict]: A list of data configurations, where each
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
        yaml_config: The source YAML configuration containing each
            a split with transforms with parameters lists

    Returns:
        list[YamlSubConfigTransformDict]: A list of data configurations, where each
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
) -> None:
    """Dumps YAML configurations to files with consistent formatting."""
    
    class CleanDumper(yaml.SafeDumper):
        """Simplified dumper maintaining key functionality"""
        def ignore_aliases(self, _data: Any) -> bool:
            return True  # Disable anchor/alias generation

        def write_line_break(self, data=None):
            """Maintain root-level spacing"""
            super().write_line_break(data)
            if not self.indent:  # At root level
                super().write_line_break()

    # Register type handlers
    CleanDumper.add_representer(type(None), 
        lambda d, _: d.represent_scalar('tag:yaml.org,2002:null', ''))
    
    CleanDumper.add_representer(list, 
        lambda d, data: d.represent_sequence(
            'tag:yaml.org,2002:seq', data, 
            flow_style=isinstance(data[0], (list, dict)) if data else False
        ))

    for i, yaml_dict in enumerate(yaml_list):
        data = _clean_params(yaml_dict.model_dump(exclude_none=True))
        
        with open(f"{directory_path}/{base_name}_{i}.yaml", "w") as f:
            yaml.dump(
                data,
                f,
                Dumper=CleanDumper,
                sort_keys=False,
                indent=2,
                width=float("inf"),
                default_flow_style=None  # Let representers handle flow style
            )

def _clean_params(data: dict) -> dict:
    """Recursive cleaner for empty parameters (replaces fix_params)"""
    if isinstance(data, dict):
        return {
            k: _clean_params(
                v if k not in ('encoder', 'transformations') 
                else [dict(e, params=e.get('params') or {}) for e in v]
            )
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_clean_params(item) for item in data]
    return data