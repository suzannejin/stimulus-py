"""Test the check_model CLI."""

import os
import warnings
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


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Setup and teardown Ray for all tests in this module."""
    # Filter ResourceWarning during Ray operations
    warnings.filterwarnings("ignore", category=ResourceWarning)
    
    # Initialize Ray with minimal resources for testing
    ray.init(ignore_reinit_error=True)
    
    yield
    
    # Ensure Ray is shut down properly after all tests
    if ray.is_initialized():
        ray.shutdown()
        
    # Clean up any ray files/directories that may have been created
    ray_results_dir = os.path.expanduser("~/ray_results")
    if os.path.exists(ray_results_dir):
        try:
            import shutil
            shutil.rmtree(ray_results_dir)
        except (PermissionError, OSError) as e:
            warnings.warn(f"Could not remove Ray results directory: {e}")


def test_check_model_main(
    data_path: str,
    data_config: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test that check_model.main runs without errors.

    Args:
        data_path: Path to test CSV data
        data_config: Path to data config YAML
        model_path: Path to model implementation
        model_config: Path to model config YAML
    """
    # Verify all required files exist
    assert os.path.exists(data_path), f"Data file not found at {data_path}"
    assert os.path.exists(data_config), f"Data config not found at {data_config}"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.exists(model_config), f"Model config not found at {model_config}"

    # Run main function - should complete without errors
    check_model.check_model(
        model_path=model_path,
        data_path=data_path,
        data_config_path=data_config,
        model_config_path=model_config,
        initial_weights=None,
        num_samples=1,
        ray_results_dirpath=None,
    )


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
