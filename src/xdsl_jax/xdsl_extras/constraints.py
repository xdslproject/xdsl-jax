"""This module contains additional type and attribute constraints that are currently not
available upstream in xDSL."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeVar

from xdsl.dialects.builtin import TupleType
from xdsl.ir import Attribute
from xdsl.irdl import (
    AttrConstraint,
    ConstraintContext,
    IntConstraint,
    IRDLAttrConstraint,
    irdl_to_attr_constraint,
)
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True, init=False)
class NestedTupleOfConstraint(AttrConstraint[TupleType]):
    """Constrain a nested tuple whose flattened leaves all match the given
    constraint.
    """

    elem_constraint: AttrConstraint

    def __init__(self, elem_constraint: IRDLAttrConstraint):
        constraint = irdl_to_attr_constraint(elem_constraint)
        object.__setattr__(self, "elem_constraint", constraint)

    def get_flattened(self, a: Attribute) -> Iterator[Attribute]:
        """Get the flattened leaves of a tuple."""
        if isinstance(a, TupleType):
            for t in a.types.data:
                yield from self.get_flattened(t)
        else:
            yield a

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        """Verify that the attribute is a tuple whose flattened leaves match the
        constraint.
        """
        if not isinstance(attr, TupleType):
            raise VerifyException(f"expected TupleType, got {type(attr)}")

        leaves: list[Attribute] = list(self.get_flattened(attr))

        for i, leaf in enumerate(leaves):
            try:
                self.elem_constraint.verify(leaf, constraint_context)
            except VerifyException as e:
                raise VerifyException(
                    f"tuple leaf {i} failed constraint: {leaf}"
                ) from e

    def mapping_type_vars(
        self,
        type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint],
    ) -> AttrConstraint[TupleType]:
        """Map type variables to constraints."""
        return NestedTupleOfConstraint(
            self.elem_constraint.mapping_type_vars(type_var_mapping)
        )
