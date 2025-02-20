"""Typing for Stimulus Python API.

This module contains all Stimulus types which will be used for variable typing
and likely not instantiated, as well as aliases for other types to use for typing purposes.

The aliases from this module should be used for typing purposes only.
"""
# ruff: noqa: F401

from typing import TypeAlias

# these imports mostly alias everything
from stimulus.data.data_handlers import (
    DatasetHandler,
    DatasetLoader,
    DatasetProcessor,
    TorchDataset,
)
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
from stimulus.learner.predict import PredictWrapper
from stimulus.learner.raytune_learner import CheckpointDict, TuneModel, TuneWrapper
from stimulus.learner.raytune_parser import (
    RayTuneMetrics,
    RayTuneOptimizer,
    RayTuneResult,
    TuneParser,
)
from stimulus.utils.performance import Performance
from stimulus.utils.yaml_model_schema import (
    CustomTunableParameter,
    Data,
    Loss,
    Model,
    RayConfigLoader,
    RayTuneModel,
    RunParams,
    Scheduler,
    TunableParameter,
    Tune,
    TuneParams,
)

# learner/raytune_parser.py

RayTuneData: TypeAlias = RayTuneMetrics | RayTuneOptimizer | RayTuneResult

# utils/yaml_data.py

Data: TypeAlias = (
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
)
