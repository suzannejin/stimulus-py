"""Tests for the transform CLI command."""

import os
import pathlib
import tempfile

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from stimulus.cli.transform import transform


# Fixtures
@pytest.fixture
def parquet_path() -> str:
    """Fixture that returns the path to a Parquet file."""
    return str(
        pathlib.Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus.parquet",
    )


@pytest.fixture
def yaml_path() -> str:
    """Fixture that returns the path to a YAML config file."""
    return str(
        pathlib.Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_transform.yaml",
    )


@pytest.mark.skip(reason="Non deterministic snapshot based on platform")
def test_transform(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Tests the transform function with correct YAML files."""
    # Verify required files exist
    assert os.path.exists(parquet_path), f"Parquet file not found at {parquet_path}"
    assert os.path.exists(yaml_path), f"YAML config not found at {yaml_path}"

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "output_dataset")

        # Run main function - should complete without errors
        transform(
            data_path=parquet_path,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output directory exists and has content
        assert os.path.exists(output_path), "Output directory was not created"
        # TODO: Add more robust verification (e.g. load dataset and check content)


@pytest.mark.skip(reason="Break github action runners")
def test_cli_invocation(
    parquet_path: str,
    yaml_path: str,
) -> None:
    """Test the CLI invocation of transform command."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = "output_dataset"
        result = runner.invoke(
            cli,
            [
                "transform",
                "-d",
                parquet_path,
                "-y",
                yaml_path,
                "-o",
                output_path,
            ],
        )
        assert result.exit_code == 0
        assert os.path.exists(output_path), "Output directory was not created"
