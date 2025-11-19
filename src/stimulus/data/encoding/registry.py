"""A registry object for Encoder objects.

This registry can add a class to the registry either by using
metaclass=EncoderRegistry in a new object or using the wrapper
function as a header `@EncoderRegistry.register(name)`
"""

from typing import ClassVar

from src.stimulus.core.registry import AbstractRegistry
from src.stimulus.data.encoding.encoders import AbstractEncoder


class EncoderRegistry(AbstractRegistry):
    """A specific registry for splitters.

    ABC meta is needed here as AbstractRegistry implements ABC and uses @abstractmethod
    """

    _REGISTRY: ClassVar[dict[str, AbstractEncoder]] = {}
    _CLASS_TYPE: object = AbstractEncoder

    @classmethod
    def save_class(
        cls,
        class_to_register: AbstractEncoder,
        name: str | None = None,
    ) -> None:
        """Checks for the right class and saves it if class correct."""
        cls._check_class(class_to_register)
        super().save_class(class_to_register=class_to_register, name=name)

    @classmethod
    def _check_class(cls, class_to_register: AbstractEncoder) -> None:
        # Check if class_to_register is a type
        if not isinstance(class_to_register, type):
            raise TypeError(f"{class_to_register} must be a class, not an instance")

        if not issubclass(type(class_to_register), type(cls._CLASS_TYPE)):
            raise TypeError(
                f"{cls._CLASS_TYPE} must be subclass of {class_to_register}",
            )
