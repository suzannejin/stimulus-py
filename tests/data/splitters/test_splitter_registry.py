"""Test for the Splitter Registry class."""

import pytest

from src.stimulus.core.registry import BaseRegistry
from src.stimulus.data.splitting import AbstractSplitter
from src.stimulus.data.splitters.registry import SplitterRegistry
from tests.core import mock


class SplitterClass(AbstractSplitter, metaclass=SplitterRegistry):
    """Base Splitter test class."""

    def __init__(self) -> None:
        """Initializes the base splitter class that should be registered for the metaclass."""
        self.base: str = "class"
        self.name: str = "splitter"

    def echo(self) -> str:
        """Returns the base + _ + name attributes."""
        return self.base + "_" + self.name


# Defining a class that inherits from base_test_class.
# It will also be added to the registry automatically due
# to its inheritance
class SplitterClass1(SplitterClass):
    """Child class of SplitterTestClass, registered through it's inheritance."""

    def __init__(self) -> None:
        """Init the parent class to get its attributes and the child class."""
        super().__init__()
        self.name = "splitter_1"


@SplitterRegistry.register("test_splitTer")
class SplitterClass2(AbstractSplitter):
    """A class that is manually registered in the registry with the header."""

    def __init__(self) -> None:
        """Defines only a name attribute."""
        self.name: str = "splitter_2"

    def echo(self) -> str:
        """Returns the name attribute."""
        return self.name


def test_num_splitter_in_registry() -> None:
    """Checks that SplitterClass, SplitterClass1, SplitterClass2."""
    splitter_registry_classes: dict[str,
                                    AbstractSplitter] = SplitterRegistry.all()
    assert len(splitter_registry_classes) == 4


def test_wrong_model_registering() -> None:
    with pytest.raises(TypeError, match=".* must be subclass of"):

        @SplitterRegistry.register("false_splitter")
        class FalseSplitter:
            def __init__():
                pass
