"""A common basic registry object for all type of classes.

This registry can add a class to the registry either by using
metaclass=BaseRegistry in a new object or using the wrapper
function as a header `@BaseRegistry.register(name)`
"""

from typing import Callable, ClassVar
from abc import ABC, abstractmethod


class AbstractRegistry(ABC):
    """Abstract class for registry.

    A registry registers and manage classes.

    Methods:
        register: wrapper to register a class.
        get: returns the class by given name.
        all: returns all registered classes.
    """

    @classmethod
    def save_class(
        cls,
        class_to_register: type[object],
        name: str | None = None,
    ) -> None:
        """Saves the class and the given name in the registry."""
        if name is None:
            name = class_to_register.__name__
        cls._REGISTRY[name.lower()] = class_to_register

    @classmethod
    def register(
        cls,
        name: str | None = None,
    ) -> Callable[[type[object]], type[object]]:
        """Function using a wrapper to register the given class with a specific name."""

        def wrapper(wrapped_class: type[object]) -> type[object]:
            """Wrapper function to save a class to the registry."""
            cls.save_class(wrapped_class, name)
            return wrapped_class

        return wrapper

    @classmethod
    def get(cls, name: str) -> type[object] | None:
        """Returns the saved classe with a given name as key."""
        return cls._REGISTRY.get(name.lower())

    @classmethod
    def all(cls) -> dict[str, type[object]]:
        """Return all the saved classes."""
        return cls._REGISTRY


class BaseRegistry(AbstractRegistry, type):
    """A generic registry implementation.

    type is necessary to be registered as a metaclass.

    # registry-subclasses.
    Source: https: // charlesreid1.github.io/python-patterns-the-registry.html
    """

    _REGISTRY: ClassVar[dict[str, type[object]]] = {}

    def __new__(cls, name: str, bases: tuple, attrs: dict[str, str]) -> type[object]:
        """Using __new__ instead of __init__ to register at definition and not class declaration.

        When a class inherits from this class as metaclass it is saved in the
        registry and all children will also be saved.

        Args:
            name(: obj: `str`, optional): A name for the registered class .
                If None, the class name will be used.
            bases(tuple): ???.
            attrs dict[str, str]: object attributes dictionnary.

        Return:
            type[object]: returns the given class as input.
        """
        new_cls: type[object] = type.__new__(cls, name, bases, attrs)
        cls.save_class(new_cls, name)
        return new_cls
