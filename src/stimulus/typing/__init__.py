"""Typing for Stimulus Python API.

This module contains all Stimulus types which will be used for variable typing
and likely not instantiated, as well as aliases for other types to use for typing purposes.

The aliases from this module should be used for typing purposes only.
"""
# ruff: noqa: F401
# ruff: noqa: F811

from typing import Any, TypeAlias, TypeVar

# these imports mostly alias everything
# Note: DatasetHandler, DatasetLoader, DatasetProcessor, TorchDataset removed as they were unused
from stimulus.data.encoding.encoders import AbstractEncoder as Encoder
from stimulus.data.interface.data_config_parser import (
    create_encoders,
    create_splitter,
    create_transforms,
)
from stimulus.data.interface.data_config_schema import (
    Columns,
    ColumnsEncoder,
    ConfigDict,
    GlobalParams,
    Schema,
    Split,
    SplitConfigDict,
    SplitTransformDict,
    Transform,
    TransformColumns,
    TransformColumnsTransformation,
)
from stimulus.data.splitting import AbstractSplitter as Splitter
from stimulus.data.transforming.transforms import AbstractTransform as Transform
from stimulus.typing.protocols import StimulusModel

# Note: PredictWrapper and Performance imports removed due to missing modules
# Note: yaml_model_schema imports removed as they contained unused Ray-related classes
# The codebase now uses Optuna exclusively for hyperparameter tuning

# data/interface/data_config_schema.py

YamlData: TypeAlias = (
    Columns
    | ColumnsEncoder
    | ConfigDict
    | GlobalParams
    | Schema
    | Split
    | SplitConfigDict
    | Transform
    | TransformColumns
    | TransformColumnsTransformation
    | SplitTransformDict
)

# Replace these problematic imports
# Note: raytune_learner imports removed as the module was replaced with optuna_tune

# Replace with type aliases if needed
CheckpointDict = dict[str, Any]
TuneModel = TypeVar("TuneModel")
TuneWrapper = TypeVar("TuneWrapper")
