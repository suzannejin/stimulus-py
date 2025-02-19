import stimulus.data.encoding.encoders as encoders
import stimulus.data.splitting.splitters as splitters
import stimulus.data.transforming.transforms as transforms
import stimulus.utils.yaml_data_schema as yaml_data_schema

def parse_config(config: yaml_data_schema.SplitTransformDict) -> tuple[
    dict[str, encoders.AbstractEncoder],
    dict[list[str], list[transforms.AbstractTransform]],
    splitters.AbstractSplitter,
]:
    """Parse the configuration and return a dictionary of the parsed configuration.
    
    Args:
        config: The configuration to parse.

    Returns:
        A tuple of the parsed configuration.
    """

    pass




