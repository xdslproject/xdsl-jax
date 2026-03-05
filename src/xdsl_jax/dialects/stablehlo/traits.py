"""
Traits specific to the StableHLO dialect.
"""

from typing import cast

from xdsl.dialects.builtin import DYNAMIC_INDEX, ShapedType, TensorType
from xdsl.ir import Attribute, Operation
from xdsl.traits import ConditionallySpeculatable, OpTrait, RecursivelySpeculatable
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.type import get_element_type_or_self, have_compatible_shape


def _is_compatible_element_type(lhs: Attribute, rhs: Attribute) -> bool:
    """Verify that the element types of the two types are compatible."""
    return get_element_type_or_self(lhs) == get_element_type_or_self(rhs)


def _is_compatible(lhs: Attribute, rhs: Attribute) -> bool:
    """Verify that the two types are compatible."""
    if isinstance(lhs, ShapedType) and isinstance(rhs, ShapedType):
        return have_compatible_shape(lhs, rhs) and _is_compatible_element_type(lhs, rhs)
    return _is_compatible_element_type(lhs, rhs)


def _static_output_dim_requires_static_input(op: Operation) -> bool:
    """Verify that the static output dimension requires the static input dimension."""
    if not op.operands or not op.results:
        return False
    input_type = cast(TensorType, op.operand_types[0])
    result_type = cast(TensorType, op.result_types[0])

    input_shape = input_type.get_shape()
    result_shape = result_type.get_shape()
    if len(input_shape) != len(result_shape):
        return False

    for idx, result_dim in enumerate(result_shape):
        if result_dim != DYNAMIC_INDEX and input_shape[idx] == DYNAMIC_INDEX:
            return False
    return True


class CompatibleOperandsAndResultType(OpTrait):
    def verify(self, op: Operation) -> None:
        """Verify that the operation has compatible types for all operands/results."""
        expected: Attribute | None = None
        if op.results:
            expected = op.result_types[0]
        if op.operands:
            expected = op.operand_types[0]
        if expected is None:
            raise VerifyException(
                f"'{op.name}' requires at least one operand or result for "
                "CompatibleOperandsAndResultType"
            )

        all_types = tuple(op.operand_types) + tuple(op.result_types)
        for actual in all_types:
            if not _is_compatible(actual, expected):
                raise VerifyException(
                    f"'{op.name}' requires compatible types for all operands/results"
                )


class RecursivelySpeculatableIfStaticDimInOutputIsStaticInInput(
    ConditionallySpeculatable
):
    @classmethod
    def is_speculatable(cls, op: Operation):
        return _static_output_dim_requires_static_input(
            op
        ) and RecursivelySpeculatable.is_speculatable(op)


class SpeculatableIfStaticDimInOutputIsStaticInInput(ConditionallySpeculatable):
    @classmethod
    def is_speculatable(cls, op: Operation):
        return _static_output_dim_requires_static_input(op)


class SpeculatableIfAllInputsStatic(ConditionallySpeculatable):
    @classmethod
    def is_speculatable(cls, op: Operation):
        return all(
            isinstance(operand_type, TensorType) and operand_type.has_static_shape()
            for operand_type in op.operand_types
