"""Test for the Transform Registry class."""

from typing import Any

import pytest

from src.stimulus.core.registry import AbstractRegistry, BaseRegistry
from src.stimulus.data.transform.registry import TransformRegistry
from src.stimulus.data.transforming.transforms import AbstractTransform
from tests.core import mock  # noqa: F401


class TransformClass(AbstractTransform, metaclass=TransformRegistry):
    """Base Transform test class."""

    def __init__(self) -> None:
        """Initializes the base transform class that should be registered for the metaclass."""
        self.base: str = "class"
        self.name: str = "transform"

    def echo(self) -> str:
        """Returns the base + _ + name attributes."""
        return self.base + "_" + self.name


# Defining a class that inherits from base_test_class.
# It will also be added to the registry automatically due
# to its inheritance
class TransformClass1(TransformClass):
    """Child class of TransformTestClass, registered through it's inheritance."""

    def __init__(self) -> None:
        """Init the parent class to get its attributes and the child class."""
        super().__init__()
        self.name = "transform_1"


@TransformRegistry.register("test_transform")
class TransformClass2(AbstractTransform):
    """A class that is manually registered in the registry with the header."""

    def __init__(self) -> None:
        """Defines only a name attribute."""
        self.name: str = "transform_2"

    def echo(self) -> str:
        """Returns the name attribute."""
        return self.name


def test_num_transform_in_registry() -> None:
    """Checks that TransformClass, TransformClass1, TransformClass2."""
    transform_registry_classes: dict[str, AbstractTransform] = TransformRegistry.all()
    assert len(transform_registry_classes) == 4


# We intentionnaly use the wrong type
def test_wrong_model_registering() -> None:
    """Checks that the model raisses an error on wrong class registering."""
    with pytest.raises(TypeError, match=".* must be subclass of"):

        @TransformRegistry.register("false_transform")  # noqa: arg-type
        class FalseTransform:
            """A class that should raise an error because it is not inheriting AbstractTransform."""

            def __init__(self):
                pass


def test_uniqueness_registry() -> None:
    """Checks that registering in the TransformRegistry doesn't save in the BaseRegistry."""
    transform_registry_classes: dict[str, AbstractTransform] = TransformRegistry.all()
    base_registry_classes: dict[str, Any] = BaseRegistry.all()
    abstract_registry_classes: dict[str, Any] = AbstractRegistry.all()

    # Generate set to get unique objects and do faster comparison
    transform_registry_set: set[AbstractTransform] = set(
        transform_registry_classes.values(),
    )
    base_registry_set: set[Any] = set(base_registry_classes.values())
    abstract_registry_set: set[Any] = set(abstract_registry_classes.values())

    # Check that there's no intersection between them
    assert len(transform_registry_set.intersection(base_registry_set)) == 0
    assert len(transform_registry_set.intersection(abstract_registry_set)) == 0
    assert len(base_registry_set.intersection(abstract_registry_set)) == 0
