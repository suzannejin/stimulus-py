"""Tests for CSV data loading and processing functionality."""

import logging
from typing import Any

import pytest
import torch

from stimulus.data.data_handlers import (
    DatasetLoader,
    DatasetProcessor,
    TorchDataset,
)
from stimulus.data.encoding import encoders as encoders_module
from stimulus.data.splitting import splitters as splitters_module
from stimulus.data.transforming import transforms as transforms_module

logger = logging.getLogger(__name__)


# Fixtures
# Data fixtures
@pytest.fixture
def titanic_csv_path() -> str:
    """Get path to test Titanic CSV file.

    Returns:
        str: Path to test CSV file
    """
    return "tests/test_data/titanic/titanic_stimulus.csv"


@pytest.fixture
def ibis_znf395_csv_path() -> str:
    """Get path to test ibis_znf395 CSV file.

    Returns:
        str: Path to test CSV file
    """
    return "tests/test_data/ibis_znf395/ibis_znf395.csv"


# Updated component fixtures
@pytest.fixture
def dummy_encoders() -> dict[str, Any]:
    """Create simple encoders for test columns."""
    return {
        "age": encoders_module.NumericEncoder(dtype=torch.float32),
        "fare": encoders_module.NumericEncoder(dtype=torch.float32),
        "survived": encoders_module.NumericEncoder(dtype=torch.int64),
    }


@pytest.fixture
def dummy_transforms() -> dict[str, list]:
    """Create test transforms."""
    return {
        "age": [
            transforms_module.GaussianNoise(std=0.1),
            transforms_module.GaussianNoise(std=0.2),
            transforms_module.GaussianNoise(std=0.3),
        ],
        "fare": [transforms_module.GaussianNoise(std=0.1)],
    }


@pytest.fixture
def ibis_znf395_transforms() -> dict[str, list]:
    """Create test transforms."""
    return {
        "dna": [transforms_module.ReverseComplement()],
    }


@pytest.fixture
def dummy_splitter() -> splitters_module.AbstractSplitter:
    """Create test splitter."""
    return splitters_module.RandomSplit(split=[0.7, 0.2, 0.1])


# Test DatasetProcessor
def test_dataset_processor_init(
    titanic_csv_path: str,
    dummy_transforms: dict,
    dummy_splitter: Any,
) -> None:
    """Test initialization of DatasetProcessor."""
    processor = DatasetProcessor(
        csv_path=titanic_csv_path,
        transforms=dummy_transforms,
        splitter=dummy_splitter,
        split_columns=["age", "fare"],
    )

    assert processor.data.shape[0] > 0
    assert isinstance(processor.transforms, dict)
    assert isinstance(processor.splitter, splitters_module.RandomSplit)
    assert isinstance(processor.transforms["age"][0], transforms_module.GaussianNoise)


def test_dataset_processor_apply_split(
    titanic_csv_path: str,
    dummy_transforms: dict,
    dummy_splitter: Any,
) -> None:
    """Test applying splits in DatasetProcessor."""
    processor = DatasetProcessor(
        csv_path=titanic_csv_path,
        transforms=dummy_transforms,
        splitter=dummy_splitter,
        split_columns=["age", "fare"],
    )

    # Should start without split column
    assert "split" not in processor.data.columns

    processor.add_split()

    assert "split" in processor.data.columns
    assert set(processor.data["split"].unique().to_list()).issubset({0, 1, 2})


def test_dataset_processor_apply_transformation_group(
    titanic_csv_path: str,
    dummy_transforms: dict,
    dummy_splitter: Any,
) -> None:
    """Test applying transformation groups."""
    processor = DatasetProcessor(
        csv_path=titanic_csv_path,
        transforms=dummy_transforms,
        splitter=dummy_splitter,
        split_columns=["age", "fare"],
    )

    control = DatasetProcessor(
        csv_path=titanic_csv_path,
        transforms={},  # No transforms
        splitter=dummy_splitter,
        split_columns=["age", "fare"],
    )

    processor.apply_transformations()

    # Transformed columns should differ
    assert processor.data["age"].to_list() != control.data["age"].to_list()
    assert processor.data["fare"].to_list() != control.data["fare"].to_list()
    # Untransformed columns should match
    assert processor.data["survived"].to_list() == control.data["survived"].to_list()


def test_dataset_processor_apply_transformation_group_ibis_znf395(
    ibis_znf395_csv_path: str,
    ibis_znf395_transforms: dict,
    dummy_splitter: Any,
) -> None:
    """Test applying transformation groups."""
    processor = DatasetProcessor(
        csv_path=ibis_znf395_csv_path,
        transforms=ibis_znf395_transforms,
        splitter=dummy_splitter,
        split_columns=["dna"],
    )

    control = DatasetProcessor(
        csv_path=ibis_znf395_csv_path,
        transforms={},
        splitter=dummy_splitter,
        split_columns=["dna"],
    )

    processor.apply_transformations()

    # Transformed columns should differ
    assert processor.data["dna"].to_list() != control.data["dna"].to_list()
    assert len(processor.data) == len(control.data) * 2


def test_dataset_processor_shuffle_labels(
    titanic_csv_path: str,
    dummy_transforms: dict,
    dummy_splitter: Any,
) -> None:
    """Test shuffling labels."""
    processor = DatasetProcessor(
        csv_path=titanic_csv_path,
        transforms=dummy_transforms,
        splitter=dummy_splitter,
        split_columns=["age", "fare"],
    )

    control = DatasetProcessor(
        csv_path=titanic_csv_path,
        transforms=dummy_transforms,
        splitter=dummy_splitter,
        split_columns=["age", "fare"],
    )
    processor.shuffle_labels(label_columns=["survived"])

    assert processor.data["survived"].to_list() != control.data["survived"].to_list()


# Test DatasetLoader
def test_dataset_loader_init(
    titanic_csv_path: str,
    dummy_encoders: dict,
) -> None:
    """Test initialization of DatasetLoader."""
    loader = DatasetLoader(
        encoders=dummy_encoders,
        input_columns=["age", "fare"],
        label_columns=["survived"],
        meta_columns=["passenger_id"],
        csv_path=titanic_csv_path,
    )

    assert loader.data.shape[0] > 0
    assert isinstance(loader.encoders, dict)
    assert len(loader.input_columns) == 2


def test_dataset_loader_get_dataset(
    titanic_csv_path: str,
    dummy_encoders: dict,
) -> None:
    """Test getting dataset from loader."""
    loader = DatasetLoader(
        encoders=dummy_encoders,
        input_columns=["age", "fare"],
        label_columns=["survived"],
        meta_columns=["passenger_id"],
        csv_path=titanic_csv_path,
    )

    inputs, labels, meta = loader.get_all_items()
    assert isinstance(inputs, dict)
    assert isinstance(labels, dict)
    assert isinstance(meta, dict)
    assert len(inputs["age"]) == len(loader.data)

    # Test encoder types
    assert isinstance(loader.encoders["age"], encoders_module.NumericEncoder)
    assert isinstance(loader.encoders["survived"], encoders_module.NumericEncoder)


def test_torch_dataset_init(
    titanic_csv_path: str,
    dummy_encoders: dict,
) -> None:
    """Test initialization of TorchDataset."""
    loader = DatasetLoader(
        encoders=dummy_encoders,
        input_columns=["age", "fare"],
        label_columns=["survived"],
        meta_columns=["passenger_id"],
        csv_path=titanic_csv_path,
    )
    dataset = TorchDataset(loader)
    assert len(dataset) > 0
    assert isinstance(dataset[0], tuple)
    assert len(dataset[0]) == 3
    assert isinstance(dataset[0][0], dict)
    assert isinstance(dataset[0][1], dict)
    assert isinstance(dataset[0][2], dict)


def test_torch_dataset_get_item(
    titanic_csv_path: str,
    dummy_encoders: dict,
) -> None:
    """Test getting item from TorchDataset."""
    loader = DatasetLoader(
        encoders=dummy_encoders,
        input_columns=["age", "fare"],
        label_columns=["survived"],
        meta_columns=["passenger_id"],
        csv_path=titanic_csv_path,
    )
    dataset = TorchDataset(loader)

    inputs, labels, meta = dataset[0]
    assert isinstance(inputs, dict)
    assert isinstance(labels, dict)
    assert isinstance(meta, dict)
    assert isinstance(inputs["age"], torch.Tensor)
    assert isinstance(inputs["fare"], torch.Tensor)
    assert isinstance(labels["survived"], torch.Tensor)
