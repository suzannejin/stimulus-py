"""Tests for the transform_csv CLI command."""

import hashlib
import os
import pathlib
import tempfile
from typing import Any, Callable

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from stimulus.cli.transform_csv import main


# Fixtures
@pytest.fixture
def csv_path() -> str:
    """Fixture that returns the path to a CSV file."""
    return str(
        pathlib.Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus.csv",
    )


@pytest.fixture
def yaml_path() -> str:
    """Fixture that returns the path to a YAML config file."""
    return str(
        pathlib.Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_transform.yaml",
    )


@pytest.mark.skip(reason="Non deterministic snapshot based on platform")
def test_transform_csv(
    csv_path: str,
    yaml_path: str,
    snapshot: Callable[[], Any],
) -> None:
    """Tests the transform_csv function with correct YAML files."""
    # Verify required files exist
    assert os.path.exists(csv_path), f"CSV file not found at {csv_path}"
    assert os.path.exists(yaml_path), f"YAML config not found at {yaml_path}"

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        # Run main function - should complete without errors
        main(
            data_csv=csv_path,
            config_yaml=yaml_path,
            out_path=output_path,
        )

        # Verify output file exists and has content
        assert os.path.exists(output_path), "Output file was not created"
        with open(output_path, newline="", encoding="utf-8") as file:
            hash = hashlib.md5(file.read().encode()).hexdigest()  # noqa: S324
        assert hash == snapshot

    finally:
        # Clean up temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.skip(reason="Break github action runners")
def test_cli_invocation(
    csv_path: str,
    yaml_path: str,
) -> None:
    """Test the CLI invocation of transform-csv command."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = "output.csv"
        result = runner.invoke(
            cli,
            [
                "transform-csv",
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
