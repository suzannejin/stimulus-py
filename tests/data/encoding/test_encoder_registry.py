"""Test for the Encoder Registry class."""

from typing import Any

import pytest

from src.stimulus.core.registry import AbstractRegistry, BaseRegistry
from src.stimulus.data.encoders.registry import EncoderRegistry
from src.stimulus.data.encoding.encoders import AbstractEncoder
from tests.core import mock  # noqa: F401


class EncoderClass(AbstractEncoder, metaclass=EncoderRegistry):
    """Base Encoder test class."""

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
class EncoderClass1(EncoderClass):
    """Child class of EncoderTestClass, registered through it's inheritance."""

    def __init__(self) -> None:
        """Init the parent class to get its attributes and the child class."""
        super().__init__()
        self.name = "splitter_1"


@EncoderRegistry.register("test_splitTer")
class EncoderClass2(AbstractEncoder):
    """A class that is manually registered in the registry with the header."""

    def __init__(self) -> None:
        """Defines only a name attribute."""
        self.name: str = "splitter_2"

    def echo(self) -> str:
        """Returns the name attribute."""
        return self.name


def test_num_splitter_in_registry() -> None:
    """Checks that EncoderClass, EncoderClass1, EncoderClass2."""
    splitter_registry_classes: dict[str, AbstractEncoder] = EncoderRegistry.all()
    assert len(splitter_registry_classes) == 4


# We intentionnaly use the wrong type
def test_wrong_model_registering() -> None:
    """Checks that the model raisses an error on wrong class registering."""
    with pytest.raises(TypeError, match=".* must be subclass of"):

        @EncoderRegistry.register("false_splitter")  # noqa: arg-type
        class FalseEncoder:
            """A class that should raise an error because it is not inheriting AbstractEncoder."""

            def __init__(self):
                pass


def test_uniqueness_registry() -> None:
    """Checks that registering in the EncoderRegistry doesn't save in the BaseRegistry."""
    splitter_registry_classes: dict[str, AbstractEncoder] = EncoderRegistry.all()
    base_registry_classes: dict[str, Any] = BaseRegistry.all()
    abstract_registry_classes: dict[str, Any] = AbstractRegistry.all()

    # Generate set to get unique objects and do faster comparison
    splitter_registry_set: set[AbstractEncoder] = set(
        splitter_registry_classes.values(),
    )
    base_registry_set: set[Any] = set(base_registry_classes.values())
    abstract_registry_set: set[Any] = set(abstract_registry_classes.values())

    # Check that there's no intersection between them
    assert len(splitter_registry_set.intersection(base_registry_set)) == 0
    assert len(splitter_registry_set.intersection(abstract_registry_set)) == 0
    assert len(base_registry_set.intersection(abstract_registry_set)) == 0
