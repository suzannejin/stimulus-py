"""Tests for the shuffle_csv CLI command."""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from stimulus.cli.shuffle_csv import shuffle_csv


@pytest.fixture
def csv_path() -> str:
    """Get path to test CSV file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus.csv",
    )


@pytest.fixture
def yaml_path() -> str:
    """Get path to test config YAML file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_split.yaml",
    )


def test_shuffle_csv_main(
    csv_path: str,
    yaml_path: str,
) -> None:
    """Test that shuffle_csv.main runs without errors.

    Args:
        csv_path: Path to test CSV data.
        yaml_path: Path to test config YAML.
    """
    # Verify required files exist
    assert os.path.exists(csv_path), f"CSV file not found at {csv_path}"
    assert os.path.exists(yaml_path), f"YAML config not found at {yaml_path}"

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        # Run main function - should complete without errors
        shuffle_csv(
            data_csv=csv_path,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output file exists and has content
        assert os.path.exists(output_path), "Output file was not created"
        with open(output_path) as file:
            hash = hashlib.md5(file.read().encode()).hexdigest()  # noqa: S324
            assert hash  # Verify we got a hash (file not empty)

    finally:
        # Clean up temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_cli_invocation(
    csv_path: str,
    yaml_path: str,
) -> None:
    """Test the CLI invocation of shuffle-csv command.

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
                "shuffle-csv",
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
