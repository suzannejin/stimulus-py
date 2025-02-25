"""Main entry point for stimulus-py cli."""

import click
from importlib_metadata import version


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version("stimulus-py"), "-v", "--version")
def cli() -> None:
    """Stimulus is an open-science framework for data processing and model training."""


@cli.command()
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that hold all transform - split - parameter info",
)
@click.option(
    "-d",
    "--out-dir",
    type=click.Path(),
    required=False,
    default="./",
    help="The output dir where all the YAMLs are written to. Output YAML will be called split-#[number].yaml transform-#[number].yaml. Default -> ./",
)
def split_split(
    yaml: str,
    out_dir: str,
) -> None:
    """Split a YAML configuration file into multiple YAML files, each containing a unique split."""
    from stimulus.cli.split_split import split_split as split_split_func

    split_split_func(config_yaml=yaml, out_dir_path=out_dir)


@cli.command()
@click.option(
    "-j",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that hold all the transform per split parameter info",
)
@click.option(
    "-d",
    "--out-dir",
    type=click.Path(),
    required=False,
    default="./",
    help="The output dir where all the YAMLs are written to. Output YAML will be called split_transform-#[number].yaml. Default -> ./",
)
def split_transforms(
    yaml: str,
    out_dir: str,
) -> None:
    """Split a YAML configuration file into multiple YAML files, each containing a unique transform."""
    from stimulus.cli.split_transforms import split_transforms as split_transforms_func

    split_transforms_func(config_yaml=yaml, out_dir_path=out_dir)
