"""Pipeline module for splitting data."""

import logging

import yaml

from stimulus.data.interface import data_config_parser
from stimulus.data.splitting import splitters

logger = logging.getLogger(__name__)


def load_splitters_from_config_from_path(
    data_config_path: str,
) -> tuple[splitters.AbstractSplitter, list[str]]:
    """Load the data config from a path.

    Args:
        data_config_path: Path to the data config file.

    Returns:
        A tuple containing the splitter instance and split input columns.
    """
    with open(data_config_path) as file:
        data_config_dict = yaml.safe_load(file)
        data_config_obj = data_config_parser.IndividualSplitConfigDict(**data_config_dict)

    return data_config_parser.parse_individual_split_config(data_config_obj)
