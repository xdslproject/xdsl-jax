"""Tests for StableHLO data movement operations."""

from xdsl.dialects.builtin import DYNAMIC_INDEX, TensorType, i32
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.data_movement_ops import ReshapeOp


def _create_reshape_op(operand_shape: list[int], result_shape: list[int]) -> ReshapeOp:
    operand = create_ssa_value(TensorType(i32, operand_shape))
    return ReshapeOp.create(
        operands=[operand],
        result_types=[TensorType(i32, result_shape)],
    )


def test_reshape_is_speculatable_with_static_operand():
    op = _create_reshape_op([2, 3], [6])

    assert op.is_speculatable()


def test_reshape_is_not_speculatable_with_dynamic_operand():
    op = _create_reshape_op([DYNAMIC_INDEX, 3], [6])

    assert not op.is_speculatable()
