"""This module contains additional type and attribute constraints that are currently not
available upstream in xDSL."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from xdsl.dialects.builtin import TupleType
from xdsl.ir import Attribute
from xdsl.irdl import (
    AttrConstraint,
    ConstraintContext,
    IntConstraint,
    irdl_to_attr_constraint,
)
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True, init=False)
class NestedTupleOfConstraint(AttrConstraint[TupleType]):
    """Constrain a nested tuple whose flattened leaves all match any allowed
    constraints."""

    elem_constraints: tuple[AttrConstraint, ...]

    def __init__(self, elem_constraints: Sequence[object]):
        object.__setattr__(
            self,
            "elem_constraints",
            tuple(irdl_to_attr_constraint(c) for c in elem_constraints),  # pyright: ignore[reportArgumentType]
        )

    def get_flattened(self, a: Attribute) -> Iterator[Attribute]:
        """Get the flattened leaves of a tuple."""
        if isinstance(a, TupleType):
            for t in a.types.data:
                yield from self.get_flattened(t)
        else:
            yield a

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        """Verify that the attribute is a tuple of allowed types."""
        if not isinstance(attr, TupleType):
            raise VerifyException(f"expected TupleType, got {type(attr)}")

        leaves: list[Attribute] = list(self.get_flattened(attr))

        for i, leaf in enumerate(leaves):
            matched = False
            for constr in self.elem_constraints:
                try:
                    constr.verify(leaf, constraint_context)
                    matched = True
                    break
                except VerifyException:
                    # Try next allowed constraint
                    pass
            if not matched:
                raise VerifyException(
                    f"tuple leaf {i} failed all allowed constraints: {leaf}"
                )

    def mapping_type_vars(
        self,
        type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint],
    ) -> AttrConstraint[TupleType]:
        """Map type variables to constraints."""
        return NestedTupleOfConstraint(
            tuple(c.mapping_type_vars(type_var_mapping) for c in self.elem_constraints)
        )
