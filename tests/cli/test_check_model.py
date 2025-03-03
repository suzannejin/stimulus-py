"""Test the check_model CLI."""

import os
import shutil
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest
import ray
from click.testing import CliRunner

from stimulus.cli import check_model
from stimulus.cli.main import cli


@pytest.fixture
def data_path() -> str:
    """Get path to test data CSV file."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus_split.csv",
    )


@pytest.fixture
def data_config() -> str:
    """Get path to test data config YAML."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_unique_transform.yaml",
    )


@pytest.fixture
def model_path() -> str:
    """Get path to test model file."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model.py")


@pytest.fixture
def model_config() -> str:
    """Get path to test model config YAML."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model_cpu.yaml")


@pytest.fixture(autouse=True)
def _ray_cleanup() -> Generator[None, None, None]:
    """Per-test Ray management with parallel execution safety."""
    # Filter ResourceWarning during Ray operations
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Initialize fresh Ray instance for each test
    ray.init(ignore_reinit_error=True)

    yield

    # Forceful cleanup for CI environments
    if ray.is_initialized():
        ray.shutdown()
        import time

        time.sleep(0.5)  # Allow background processes to exit

    # Clean any residual files
    ray_results_dir = os.path.expanduser("~/ray_results")
    shutil.rmtree(ray_results_dir, ignore_errors=True)


def test_check_model_main(
    data_path: str,
    data_config: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test that check_model.main runs without errors."""
    # Verify all required files exist
    assert os.path.exists(data_path), f"Data file not found at {data_path}"
    assert os.path.exists(data_config), f"Data config not found at {data_config}"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.exists(model_config), f"Model config not found at {model_config}"

    # Run main function - should complete without errors
    try:
        check_model.check_model(
            model_path=model_path,
            data_path=data_path,
            data_config_path=data_config,
            model_config_path=model_config,
            initial_weights=None,
            num_samples=1,
            ray_results_dirpath=None,
        )
    except RuntimeError as e:
        pytest.fail(f"check_model.check_model raised {type(e).__name__}: {e}")


def test_cli_invocation(
    data_path: str,
    data_config: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test the CLI invocation of check-model command.

    Args:
        data_path: Path to test CSV data.
        data_config: Path to data config YAML.
        model_path: Path to model implementation.
        model_config: Path to model config YAML.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "check-model",
            "-d",
            data_path,
            "-m",
            model_path,
            "-e",
            data_config,
            "-c",
            model_config,
            "-n",
            "1",
        ],
    )
    assert result.exit_code == 0
