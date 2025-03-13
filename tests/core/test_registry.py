# """Tests for the BaseRegistry class."""
from src.stimulus.core.registry import BaseRegistry
from tests.core import mock

from typing import Type


# Defining a meta class that will automatically be registered
# as it inherits BaseRegistry as a metaclass.
class base_test_class(metaclass=BaseRegistry):
    def __init__(self) -> None:
        self.base: str = "class"
        self.name: str = "base"

    def echo(self):
        return self.base + "_" + self.name


# Defining a class that inherits from base_test_class.
# It will also be added to the registry automatically due
# to its inheritance
class class1(base_test_class):
    def __init__(self) -> None:
        super().__init__()
        self.name = "1"


# Register manually a class with the header
@BaseRegistry.register("test_CLASS2")
class class2:
    def __init__(self) -> None:
        self.name: str = "class_2"

    def echo(self) -> str:
        return self.name


def test_num_class_in_registry() -> None:
    """
    Checks that class1, class2 and mock.class3 are all correctly saved.
    """
    registry_classes: dict[str, Type[object]] = BaseRegistry.all()
    assert len(registry_classes) == 4  # 3 in this file + mock


def test_name_is_lower_in_registry() -> None:
    """Checks that all the names are saved to lowercase."""
    registry_classes: dict[str, Type[object]] = BaseRegistry.all()
    for name, associated_class in registry_classes.items():
        assert name.lower() == name


def test_class_init_in_registry() -> None:
    """Checks that all the class can be inited and print."""
    registry_classes: dict[str, Type[object]] = BaseRegistry.all()
    for name, associated_class in registry_classes.items():
        name_class: Type[object] = associated_class()
        assert "class" in name_class.echo()
