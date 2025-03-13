from src.stimulus.core.registry import BaseRegistry


@BaseRegistry.register("tEst_ClAss3")
class class3:
    def echo(self) -> str:
        return "class3"
