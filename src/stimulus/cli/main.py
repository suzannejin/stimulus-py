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
    "-c",
    "--csv",
    type=click.Path(exists=True),
    required=True,
    help="The file path for the csv containing the data in csv format",
)
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML data config",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="The output file path to write the shuffled csv",
)
def shuffle_csv(
    csv: str,
    yaml: str,
    output: str,
) -> None:
    """Shuffle rows in a CSV data file."""
    from stimulus.cli.shuffle_csv import shuffle_csv as shuffle_csv_func

    shuffle_csv_func(
        data_csv=csv,
        config_yaml=yaml,
        out_path=output,
    )


@cli.command()
@click.option(
    "-c",
    "--csv",
    type=click.Path(exists=True),
    required=True,
    help="The file path for the csv containing all data",
)
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that holds all parameter info",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="The output file path to write the split csv",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite the split column if it already exists in the csv",
)
def split_csv(
    csv: str,
    yaml: str,
    output: str,
    *,
    force: bool,
) -> None:
    """Split rows in a CSV data file."""
    from stimulus.cli.split_csv import split_csv as split_csv_func

    split_csv_func(
        data_csv=csv,
        config_yaml=yaml,
        out_path=output,
        force=force,
    )


@cli.command()
@click.option(
    "-y",
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
