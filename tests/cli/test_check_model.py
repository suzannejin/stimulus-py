"""Test the check_model CLI."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from stimulus.cli import check_model

logger = logging.getLogger(__name__)


@pytest.fixture
def data_path() -> str:
    """Get path to test data directory."""
    return str(
        Path(__file__).parent.parent / "test_data" / "titanic_performant" / "titanic_encoded_hf",
    )


@pytest.fixture
def model_path() -> str:
    """Get path to test model file."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_perf_model.py")


@pytest.fixture
def model_config() -> str:
    """Get path to test model config YAML."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_perf_model.yaml")


def test_check_model_main(
    data_path: str,
    model_path: str,
    model_config: str,
) -> None:
    """Test that check_model.main runs without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        assert os.path.exists(data_path), f"Data file not found at {data_path}"
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        assert os.path.exists(model_config), f"Model config not found at {model_config}"

        # Run main function - should complete without errors
        try:
            base_path, file_path = check_model.check_model(
                data_path=data_path,
                model_path=model_path,
                model_config_path=model_config,
                optuna_results_dirpath=temp_dir,
            )
        except RuntimeError as e:
            pytest.fail(f"check_model.check_model raised {type(e).__name__}: {e}")

        assert os.path.exists(base_path)
        assert os.path.exists(file_path)
        assert os.path.exists(f"{base_path}/artifacts/")
        assert os.path.exists(f"{base_path}/optuna_journal_storage.log")

        shutil.rmtree(temp_dir)
