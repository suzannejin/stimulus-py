from typing import Type, Callable


class BaseRegistry(type):
    """
    A generic registry implementation.
    source: https://charlesreid1.github.io/python-patterns-the-registry.html#registry-subclasses
    """

    _REGISTRY: dict[str, Type[object]] = {}

    def __new__(cls, name: str, bases: tuple, attrs: dict[str, str]) -> None:
        """
        Using __new__ instead of __init__ to register at definition and not class declaration.

        When a class inherits from this class as metaclass it is saved in the registry and all children will also be saved

        Args:
            class_to_register (Any): The python class to register.
            name (:obj:`str`, optional): A name for the registered class.
                If None, the class name will be used
        """
        new_cls: Type[object] = type.__new__(cls, name, bases, attrs)
        cls.save_class(new_cls, name)
        return new_cls

    @classmethod
    def save_class(
        cls, class_to_register: Type[object], name: str | None = None
    ) -> None:
        """Saves the class and the given name in the registry."""
        if name is None:
            name = class_to_register.__name__
        cls._REGISTRY[name.lower()] = class_to_register

    @classmethod
    def register(cls, name: str | None = None) -> Type[object]:
        def wrapper(wrapped_class: Type[object]) -> Type[object]:
            """Wrapper function to save a class to the registry."""
            cls.save_class(wrapped_class, name)
            return wrapped_class

        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[object]:
        """Returns the saved classe with a given name as key."""
        return cls._REGISTRY.get(name.lower())

    @classmethod
    def all(cls) -> dict[str, Type[object]]:
        """Return all the saved classes."""
        return cls._REGISTRY
