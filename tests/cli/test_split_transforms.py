"""Test for the split_transforms CLI command."""

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from src.stimulus.cli.split_transforms import split_transforms


# Fixtures
@pytest.fixture
def correct_yaml_path() -> str:
    """Fixture that returns the path to a correct YAML file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_split.yaml",
    )


@pytest.fixture
def wrong_yaml_path() -> str:
    """Fixture that returns the path to a wrong YAML file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "yaml_files" / "wrong_field_type.yaml",
    )


# Test cases
test_cases = [("correct_yaml_path", None), ("wrong_yaml_path", ValueError)]


# Tests
@pytest.mark.parametrize(("yaml_type", "error"), test_cases)
def test_split_transforms(
    request: pytest.FixtureRequest,
    snapshot: Callable[[], Any],
    yaml_type: str,
    error: Exception | None,
    tmp_path: Path,  # Pytest tmp file system
) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    yaml_path: str = request.getfixturevalue(yaml_type)
    tmpdir = str(tmp_path)
    if error is not None:
        with pytest.raises(error):  # type: ignore[call-overload]
            split_transforms(yaml_path, tmpdir)
    else:
        # convert tmpdir to str
        split_transforms(yaml_path, tmpdir)
        files = os.listdir(tmpdir)
        test_out = [f for f in files if f.startswith("test_")]
        hashes = []
        for f in test_out:
            with open(os.path.join(tmpdir, f)) as file:
                hashes.append(hashlib.sha256(file.read().encode()).hexdigest())
        assert sorted(hashes) == snapshot  # Sorted ensures that the order of the hashes does not matter


def test_cli_invocation(
    correct_yaml_path: str
) -> None:
    """Test the CLI invocation of split-split command.
    Args:
        config_yaml: Path to the YAML config file.
        out_dir: Path to the output directory.
    """
    runner = CliRunner()
    with runner.isolated_filesystem():
        output_path = tempfile.gettempdir()
        result = runner.invoke(
            cli,
            [
                "split-transforms",
                "-j",
                correct_yaml_path,
                "-d",
                output_path,
            ],
        )
        files = os.listdir(output_path)
        test_out = [f for f in files if f.startswith("test_")]
        assert result.exit_code == 0
        assert len(test_out) > 0, "No output files were generated"
