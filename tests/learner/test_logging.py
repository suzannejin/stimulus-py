"""Tests for unified experiment logging interface."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from stimulus.learner.logging import ExperimentLogger, create_trial_logger


def _is_wandb_available() -> bool:
    """Check if wandb is installed."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    else:
        return True


@pytest.fixture
def temp_log_dir() -> Generator[str, None, None]:
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestExperimentLogger:
    """Tests for ExperimentLogger class."""

    def test_tensorboard_backend(self, temp_log_dir: str) -> None:
        """Test TensorBoard backend initialization and basic logging."""
        logger = ExperimentLogger(log_dir=temp_log_dir, backend="tensorboard")

        # Log some scalars
        logger.log_scalar("train/loss", 0.5, step=0)
        logger.log_scalar("train/loss", 0.4, step=1)

        # Log multiple scalars
        logger.log_scalars({"val/acc": 0.8, "val/loss": 0.3}, step=1)

        # Close logger
        logger.close()

        # Verify event files were created
        event_files = list(Path(temp_log_dir).glob("events.out.tfevents.*"))
        assert len(event_files) > 0, "TensorBoard event files should be created"

    def test_log_hparams(self, temp_log_dir: str) -> None:
        """Test hyperparameter logging."""
        logger = ExperimentLogger(log_dir=temp_log_dir, backend="tensorboard")

        hparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "layers": [128, 64, 32],  # Complex type
            "optimizer": "Adam",
        }
        metrics = {"final_loss": 0.1, "final_acc": 0.95}

        logger.log_hparams(hparams, metrics, run_name="test_run")
        logger.close()

        # Verify event files exist
        event_files = list(Path(temp_log_dir).glob("**/events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_wandb_backend_missing_project(self, temp_log_dir: str) -> None:
        """Test that WandB backend requires project name."""
        with pytest.raises(ValueError, match="wandb_project is required"):
            ExperimentLogger(log_dir=temp_log_dir, backend="wandb")

    @pytest.mark.skipif(
        not _is_wandb_available(),
        reason="WandB not installed",
    )
    def test_wandb_backend_with_project(self, temp_log_dir: str) -> None:
        """Test WandB backend with project name."""
        # This test requires wandb to be installed
        # It will create a wandb run but in offline mode
        logger = ExperimentLogger(
            log_dir=temp_log_dir,
            backend="wandb",
            wandb_project="test-project",
            mode="offline",  # Don't sync to cloud
        )

        logger.log_scalar("train/loss", 0.5, step=0)
        logger.close()

        # Verify both TB and wandb files exist
        event_files = list(Path(temp_log_dir).glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_both_backends_missing_project(self, temp_log_dir: str) -> None:
        """Test that 'both' backend requires project name."""
        with pytest.raises(ValueError, match="wandb_project is required"):
            ExperimentLogger(log_dir=temp_log_dir, backend="both")

    def test_log_dir_creation(self) -> None:
        """Test that log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "new_logs", "experiment_1")
            assert not os.path.exists(log_dir)

            logger = ExperimentLogger(log_dir=log_dir, backend="tensorboard")
            assert os.path.exists(log_dir)
            logger.close()


class TestCreateTrialLogger:
    """Tests for create_trial_logger helper function."""

    def test_create_trial_logger(self, temp_log_dir: str) -> None:
        """Test trial logger creation."""
        trial_logger = create_trial_logger(
            root_log_dir=temp_log_dir,
            trial_number=5,
            backend="tensorboard",
        )

        # Verify log directory structure
        expected_dir = os.path.join(temp_log_dir, "trial-5")
        assert trial_logger.log_dir == expected_dir
        assert os.path.exists(expected_dir)

        trial_logger.log_scalar("train/loss", 0.3, step=0)
        trial_logger.close()

    def test_multiple_trial_loggers(self, temp_log_dir: str) -> None:
        """Test creating multiple trial loggers."""
        loggers = []
        for i in range(3):
            logger = create_trial_logger(
                root_log_dir=temp_log_dir,
                trial_number=i,
                backend="tensorboard",
            )
            logger.log_scalar("train/loss", 0.1 * i, step=0)
            loggers.append(logger)

        # Verify separate directories
        trial_dirs = list(Path(temp_log_dir).glob("trial-*"))
        assert len(trial_dirs) == 3

        for logger in loggers:
            logger.close()


class TestLoggerBackendType:
    """Tests for LoggerBackend type."""

    def test_valid_backends(self, temp_log_dir: str) -> None:
        """Test all valid backend values."""
        # Only test tensorboard (others require wandb)
        logger = ExperimentLogger(log_dir=temp_log_dir, backend="tensorboard")
        assert logger.backend == "tensorboard"
        logger.close()
