from typing import Any

from stimulus.data.encoding import encoders
from stimulus.data.splitting import splitters
from stimulus.data.transforming import transforms
from stimulus.utils import yaml_data_schema


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


def create_encoders(column_config: list[yaml_data_schema.Columns]) -> dict[str, encoders.AbstractEncoder]:
    """Factory for creating encoders from config"""
    return {
        column.column_name: _instantiate_component(
            module=encoders,
            name=column.encoder[0].name,
            params=column.encoder[0].params,
        )
        for column in column_config
    }


def create_transforms(transform_config: list[yaml_data_schema.TransformColumns]) -> dict[str, list[Any]]:
    """Factory for creating transforms from config"""
    return {
        column.column_name: [
            _instantiate_component(
                module=transforms,
                name=transformation.name,
                params=transformation.params,
            )
            for transformation in column.transformations
        ]
        for column in transform_config.columns
    }


def create_splitter(split_config: list[yaml_data_schema.Split]) -> splitters.AbstractSplitter:
    """Factory for creating splitters from config"""
    return _instantiate_component(
        module=splitters,
        name=split_config[0].split_method,
        params=split_config[0].params,
    )


def parse_config(
    config: yaml_data_schema.SplitTransformDict,
) -> tuple[
    dict[str, encoders.AbstractEncoder],
    dict[str, list[transforms.AbstractTransform]],
    splitters.AbstractSplitter,
]:
    """Parse the configuration and return a dictionary of the parsed configuration.

    Args:
        config: The configuration to parse.

    Returns:
        A tuple of the parsed configuration.
    """
    encoders = create_encoders(config.columns)
    transforms = create_transforms(config.transforms)
    splitter = create_splitter(config.split)

    return encoders, transforms, splitter
