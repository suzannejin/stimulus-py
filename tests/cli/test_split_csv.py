"""Tests for the split_csv CLI command."""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from stimulus.cli.split_csv import split_csv


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


@pytest.mark.skip(reason="Break github action runners")
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


@pytest.mark.skip(reason="Break github action runners")
def test_split_csv_with_force(
    csv_path_with_split: str,
    yaml_path: str,
) -> None:
    """Test split_csv with force flag on file that already has split column.

    Args:
        csv_path_with_split: Path to test CSV data with split column.
        yaml_path: Path to test config YAML.
    """
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        output_path = tmp_dir_name

        split_csv(
            data_csv=csv_path_with_split,
            config_yaml=yaml_path,
            out_path=output_path,
            force=True,
        )
        assert os.path.exists(output_path)


@pytest.mark.skip(reason="Break github action runners")
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
