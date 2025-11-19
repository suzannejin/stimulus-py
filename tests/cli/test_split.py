"""Tests for the split_csv CLI command."""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from stimulus.cli.split import split_csv


@pytest.fixture
def csv_path() -> str:
    """Get path to test CSV file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus.csv",
    )


@pytest.fixture
def csv_path_with_split() -> str:
    """Get path to test CSV file with split column."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus_split.csv",
    )


@pytest.fixture
def yaml_path() -> str:
    """Get path to test config YAML file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_split.yaml",
    )


# @pytest.mark.skip(reason="Break github action runners")
def test_split_csv_main(
    csv_path: str,
    yaml_path: str,
) -> None:
    """Test that split_csv runs without errors.

    Args:
        csv_path: Path to test CSV data.
        yaml_path: Path to test config YAML.
    """
    # Verify required files exist
    assert os.path.exists(csv_path), f"CSV file not found at {csv_path}"
    assert os.path.exists(yaml_path), f"YAML config not found at {yaml_path}"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        output_path = tmp_dir_name

        # Run main function - should complete without errors
        split_csv(
            data_csv=csv_path,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output directory exists and contains expected files
        assert os.path.exists(output_path), "Output directory was not created"
        assert os.path.exists(
            os.path.join(output_path, "train"),
        ), "train split directory not found"
        assert os.path.exists(
            os.path.join(output_path, "test"),
        ), "test split directory not found"


# @pytest.mark.skip(reason="Break github action runners")
def test_split_csv_error_on_existing_split(
    csv_path_with_split: str,
    yaml_path: str,
) -> None:
    """Test split_csv raises error on file that already has split column.

    Args:
        csv_path_with_split: Path to test CSV data with split column.
        yaml_path: Path to test config YAML.
    """
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        output_path = tmp_dir_name

        # First run to create the split dataset
        # Actually csv_path_with_split is just a CSV, it doesn't necessarily mean it's a HF dataset with splits.
        # But the test name implies it has a split column or something.
        # Wait, if it's a CSV, load_dataset_from_path loads it as a single split (usually 'train').
        # So split() should work fine unless we are loading a directory that is already split.

        # Let's look at how split_csv works. It loads data.
        # If data_csv is a directory, it loads from disk.
        # If it's a csv file, it loads as csv.

        # If we want to test the "already exists" error, we need to pass a dataset that has 'test' split.
        # So we should first split and save, then try to split again the SAVED output.

        # Let's adjust the test to do exactly that.

        # 1. Split and save to output_path
        split_csv(
            data_csv=csv_path_with_split,  # This is just a CSV, so it works first time
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # 2. Try to split again using the output_path as input
        with pytest.raises(ValueError, match="Test split already exists"):
            split_csv(
                data_csv=output_path,  # Now pointing to the directory with splits
                config_yaml=yaml_path,
                out_path=output_path,  # Output doesn't matter here, should fail before
            )


# @pytest.mark.skip(reason="Break github action runners")
def test_cli_invocation(
    csv_path: str,
    yaml_path: str,
) -> None:
    """Test the CLI invocation of split-csv command.

    Args:
        csv_path: Path to test CSV data.
        yaml_path: Path to test config YAML.
    """
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = "output.csv"
        result = runner.invoke(
            cli,
            [
                "split-csv",
                "-c",
                csv_path,
                "-y",
                yaml_path,
                "-o",
                output_path,
            ],
        )
        assert result.exit_code == 0
        assert os.path.exists(output_path), "Output file was not created"
