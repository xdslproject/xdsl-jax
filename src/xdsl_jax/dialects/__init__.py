from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    def get_stablehlo():
        from xdsl_jax.dialects.stablehlo import StableHLO

        return StableHLO

    return {
        "stablehlo": get_stablehlo,
    }
