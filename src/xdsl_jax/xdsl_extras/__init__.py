"""This module contains additional constraints and traits not available upstream in
xDSL. These should ideally be upstreamed to xDSL, but are not yet.
"""

from xdsl_jax.xdsl_extras.constraints import NestedTupleOfConstraint
from xdsl_jax.xdsl_extras.traits import (
    AllMatchSameOperatorTrait,
    Elementwise,
    SameOperandsAndResultElementType,
    SameOperandsAndResultShape,
    SameOperandsElementType,
)

__all__ = [
    # Constraints
    "NestedTupleOfConstraint",
    # Traits
    "AllMatchSameOperatorTrait",
    "Elementwise",
    "SameOperandsAndResultElementType",
    "SameOperandsAndResultShape",
    "SameOperandsElementType",
]
