"""A mock class to show the registering on import in tests/core/test_registry.py."""

from src.stimulus.core.registry import BaseRegistry


@BaseRegistry.register("tEst_ClAss3")
class Class3:
    """A mock class."""

    def echo(self) -> str:
        """Returns just class3."""
        return "class3"
