"""Tests for the BaseRegistry class."""

from src.stimulus.core.registry import BaseRegistry
from tests.core import mock  # noqa: F401


# Defining a meta class that will automatically be registered
# as it inherits BaseRegistry as a metaclass.
class BaseTestClass(metaclass=BaseRegistry):
    """Base test class that is registered due to the metaclass."""

    def __init__(self) -> None:
        """Initializes a base and a name attribute."""
        self.base: str = "class"
        self.name: str = "base"

    def echo(self) -> str:
        """Returns the base + _ + name attributes."""
        return self.base + "_" + self.name


# Defining a class that inherits from base_test_class.
# It will also be added to the registry automatically due
# to its inheritance
class Class1(BaseTestClass):
    """Child class of base_test_class, registered through it's inheritance."""

    def __init__(self) -> None:
        """Init the parent class to get its attributes and the child class."""
        super().__init__()
        self.name = "1"


# Register manually a class with the header
@BaseRegistry.register("test_CLASS2")
class Class2:
    """A class that is manually registered in the registry with the header."""

    def __init__(self) -> None:
        """Defines only a name attribute."""
        self.name: str = "class_2"

    def echo(self) -> str:
        """Returns the name attribute."""
        return self.name


def test_num_class_in_registry() -> None:
    """Checks that class1, class2 and mock.class3 are all correctly saved."""
    registry_classes: dict[str, type[object]] = BaseRegistry.all()
    assert len(registry_classes) == 4  # 3 in this file + mock


def test_name_is_lower_in_registry() -> None:
    """Checks that all the names are saved to lowercase."""
    registry_classes: dict[str, type[object]] = BaseRegistry.all()
    for name, _associated_class in registry_classes.items():
        assert name.lower() == name


def test_class_init_in_registry() -> None:
    """Checks that all the class can be inited and print."""
    registry_classes: dict[str, type[object]] = BaseRegistry.all()
    for _name, associated_class in registry_classes.items():
        name_class: object = associated_class()
        assert "class" in name_class.echo()  # type: ignore[attr-defined]
