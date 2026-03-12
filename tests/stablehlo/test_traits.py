"""Tests for StableHLO-specific traits."""

from typing import TypeAlias

import pytest
from xdsl.dialects.builtin import DYNAMIC_INDEX, TensorType, f32, i32
from xdsl.ir import Attribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.traits import (
    CompatibleOperandsAndResultType,
)

AnyTensorType: TypeAlias = TensorType[Attribute]


@irdl_op_definition
class CompatOp(IRDLOperation):
    name = "test.compat"
    traits = traits_def(CompatibleOperandsAndResultType())

    lhs = operand_def()
    rhs = operand_def()
    res = result_def()


@irdl_op_definition
class EmptyCompatOp(IRDLOperation):
    name = "test.empty_compat"
    traits = traits_def(CompatibleOperandsAndResultType())


def test_compatible_operands_and_result_type_pass():
    operand1 = create_ssa_value(TensorType(i32, [2, 3]))
    operand2 = create_ssa_value(TensorType(i32, [2, DYNAMIC_INDEX]))
    op = CompatOp.create(
        operands=[operand1, operand2], result_types=[TensorType(i32, [2, 3])]
    )
    op.verify()


def test_compatible_operands_and_result_type_shape_mismatch():
    operand1 = create_ssa_value(TensorType(i32, [2, 3]))
    operand2 = create_ssa_value(TensorType(i32, [3, 2]))
    op = CompatOp.create(
        operands=[operand1, operand2], result_types=[TensorType(i32, [2, 3])]
    )
    with pytest.raises(
        VerifyException, match="requires compatible types for all operands/results"
    ):
        op.verify()


def test_compatible_operands_and_result_type_element_mismatch():
    operand1 = create_ssa_value(TensorType(i32, [2]))
    operand2 = create_ssa_value(TensorType(f32, [2]))
    op = CompatOp.create(
        operands=[operand1, operand2], result_types=[TensorType(i32, [2])]
    )
    with pytest.raises(
        VerifyException, match="requires compatible types for all operands/results"
    ):
        op.verify()


def test_compatible_operands_and_result_type_requires_operand_or_result():
    op = EmptyCompatOp.create()
    with pytest.raises(
        VerifyException, match="requires at least one operand or result"
    ):
        op.verify()


def test_compatible_operands_and_result_type_accepts_matching_non_shaped_types():
    operand1 = create_ssa_value(i32)
    operand2 = create_ssa_value(i32)
    op = CompatOp.create(operands=[operand1, operand2], result_types=[i32])
    op.verify()
