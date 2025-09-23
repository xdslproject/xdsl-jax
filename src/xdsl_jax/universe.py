from xdsl.universe import Universe

from xdsl_jax.dialects import get_all_dialects
from xdsl_jax.transforms import get_all_passes

UNIVERSE = Universe(
    all_dialects=get_all_dialects(),
    all_passes=get_all_passes(),
)
