"""A mock class to show the registering on import in tests/core/test_registry.py."""

from src.stimulus.core.registry import BaseRegistry
from src.stimulus.data.encoders.registry import EncoderRegistry
from src.stimulus.data.encoding.encoders import AbstractEncoder
from src.stimulus.data.splitters.registry import SplitterRegistry
from src.stimulus.data.splitting.splitters import AbstractSplitter
from src.stimulus.data.transform.registry import TransformRegistry
from src.stimulus.data.transforming.transforms import AbstractTransform


@BaseRegistry.register("tEst_ClAss3")
class Class3:
    """A mock class."""

    def echo(self) -> str:
        """Returns just class3."""
        return "class3"


@SplitterRegistry.register("SplitTER_ClAss3")
class Splitter3(AbstractSplitter):
    """A mock class."""

    def echo(self) -> str:
        """Returns just class3."""
        return "splitter_3"


@EncoderRegistry.register("EnCOdEr_ClAss3")
class Encoder3(AbstractEncoder):
    """A mock class."""

    def echo(self) -> str:
        """Returns just class3."""
        return "encoder_3"


@TransformRegistry.register("TranSFOrM_CLASs3")
class Transform3(AbstractTransform):
    """A mock class."""

    def echo(self) -> str:
        """Returns just transform_3."""
        return "transform_3"
