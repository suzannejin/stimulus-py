"""Main entry point for stimulus-py cli."""

import click
from importlib_metadata import version


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version("stimulus-py"), "-v", "--version")
def cli() -> None:
    """Stimulus is an open-science framework for data processing and model training."""


@cli.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to input csv file",
)
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model file",
)
@click.option(
    "-e",
    "--data-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to data config file",
)
@click.option(
    "-c",
    "--model-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to yaml config training file",
)
@click.option(
    "-w",
    "--initial-weights",
    type=click.Path(exists=True),
    help="Path to initial weights",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    default=3,
    help="Number of samples for tuning [default: 3]",
)
@click.option(
    "--ray-results-dirpath",
    type=click.Path(),
    help="Location for ray_results output dir",
)
@click.option(
    "--debug-mode",
    is_flag=True,
    help="Activate debug mode for tuning",
)
def check_model(
    data: str,
    model: str,
    data_config: str,
    model_config: str,
    initial_weights: str | None,
    num_samples: int,
    ray_results_dirpath: str | None,
    *,
    debug_mode: bool,
) -> None:
    """Check model configuration and run initial tests."""
    from stimulus.cli.check_model import check_model as check_model_func

    check_model_func(
        data_path=data,
        model_path=model,
        data_config_path=data_config,
        model_config_path=model_config,
        initial_weights=initial_weights,
        num_samples=num_samples,
        ray_results_dirpath=ray_results_dirpath,
        debug_mode=debug_mode,
    )
