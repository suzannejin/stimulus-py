"""Tests for the split_split CLI command."""

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import pytest
from click.testing import CliRunner

from stimulus.cli.main import cli
from stimulus.cli import split_split


# Fixtures
@pytest.fixture
def correct_yaml_path() -> str:
    """Fixture that returns the path to a correct YAML file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic.yaml",
    )


@pytest.fixture
def wrong_yaml_path() -> str:
    """Fixture that returns the path to a wrong YAML file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "yaml_files" / "wrong_field_type.yaml",
    )


# Test cases
test_cases = [
    ("correct_yaml_path", None),
    ("wrong_yaml_path", ValueError),
]


# Tests
@pytest.mark.parametrize(("yaml_type", "error"), test_cases)
def test_split_split_main(
    yaml_type: str,
    error: Optional[Exception],
    request: Any,
    snapshot: Any
) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    yaml_path = request.getfixturevalue(yaml_type)
    tmpdir = tempfile.gettempdir()
    if error:
        with pytest.raises(error):  # type: ignore[call-overload]
            split_split.split_split(yaml_path, tmpdir)
    else:
        split_split.split_split(yaml_path, tmpdir)  # split_split() returns None, no need to assert
        files = os.listdir(tmpdir)
        test_out = [f for f in files if f.startswith("test_")]
        assert len(test_out) > 0, "No output files were generated"
        hashes = []
        for f in test_out:
            with open(os.path.join(tmpdir, f)) as file:
                hashes.append(hashlib.md5(file.read().encode()).hexdigest())  # noqa: S324
        assert sorted(hashes) == snapshot  # sorted ensures that the order of the hashes does not matter


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
                "split-split",
                "-y",
                correct_yaml_path,
                "-d",
                output_path,
            ],
        )
        files = os.listdir(output_path)
        test_out = [f for f in files if f.startswith("test_")]
        assert result.exit_code == 0
        assert len(test_out) > 0, "No output files were generated"
