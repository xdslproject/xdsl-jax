from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    def get_stablehlo():
        from xdsl.dialects.stablehlo import StableHLO

        return StableHLO

    return {
        "stablehlo2": get_stablehlo,
    }
