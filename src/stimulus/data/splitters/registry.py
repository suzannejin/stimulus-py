"""A registry object for Splitter objects.

This registry can add a class to the registry either by using
metaclass=SplitterRegistry in a new object or using the wrapper
function as a header `@SplitterRegistry.register(name)`
"""

from abc import ABCMeta
from src.stimulus.core.registry import AbstractRegistry
from src.stimulus.data.splitting.splitters import AbstractSplitter
from typing import Any, ClassVar


class SplitterRegistry(AbstractRegistry, ABCMeta):
    """
    A specific registry for splitters.

    ABC meta is needed here as AbstractRegistry implements ABC and uses @abstractmethod
    """

    _REGISTRY: ClassVar[dict[str, AbstractSplitter]] = {}
    _CLASS_TYPE: AbstractRegistry = AbstractRegistry

    @classmethod
    def _check_class(cls, class_to_register):
        # Check if class_to_register is a type
        if not isinstance(class_to_register, type):
            raise TypeError(
                f"{class_to_register} must be a class, not an instance")

        # Use try-except to handle potential errors
        try:
            if not issubclass(type(class_to_register), type(cls._CLASS_TYPE)):
                raise TypeError(
                    f"{cls._CLASS_TYPE} must be subclass of {class_to_register}"
                )
        except TypeError as e:
            print(f"Error during subclass check: {e}")
            # If there's an error, assume it's not a subclass
            raise TypeError(
                f"{class_to_register} must be subclass of \
                {cls._CLASS_TYPE.__name__}"
            )

    def __new__(
        cls, name: str, bases: tuple, attrs: dict[str, Any]
    ) -> AbstractSplitter:
        """
        cls: Class to register
        name: The name of the class that is registered (that inherits from SplitterRegistry).
        bases: The other base class from which the registered class inherits.
        attrs: The registered class attributes.
        """
        new_splitter: AbstractSplitter = type.__new__(cls, name, bases, attrs)
        cls.save_class(new_splitter, name)
        return new_splitter

    @classmethod
    def save_class(
        cls, class_to_register: AbstractSplitter, name: str | None = None
    ) -> None:
        cls._check_class(class_to_register)
        super().save_class(class_to_register=class_to_register, name=name)
