"""Main entry point for stimulus-py cli."""

import click
from importlib_metadata import version

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(version("stimulus-py"), "-v", "--version")
def cli():
    """Stimulus is an open-science framework for data processing and model training."""
    pass


@cli.command()
def check_model():
    """Check model configuration and run initial tests.

    check-model will connect to an existing ray cluster. Make sure you start a ray cluster before by running:
    ray start --head

    \b
    Required Options:
      -d, --data PATH             Path to input csv file
      -m, --model PATH            Path to model file
      -e, --data-config PATH      Path to data config file
      -c, --model-config PATH     Path to yaml config training file

    \b
    Optional Options:
      -w, --initial-weights PATH  Path to initial weights
      -n, --num-samples INTEGER   Number of samples for tuning [default: 3]
      --ray-results-dirpath PATH  Location for ray_results output dir
      --debug-mode                Activate debug mode for tuning
    """
    from stimulus.cli.check_model import main
    main()
