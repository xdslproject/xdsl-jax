"""Tests for StableHLO data movement operations."""

import pytest
from xdsl.dialects.builtin import DYNAMIC_INDEX, TensorType, i32
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.data_movement_ops import ReshapeOp


def _create_reshape_op(operand_shape: list[int], result_shape: list[int]) -> ReshapeOp:
    operand = create_ssa_value(TensorType(i32, operand_shape))
    return ReshapeOp.create(
        operands=[operand],
        result_types=[TensorType(i32, result_shape)],
    )


@pytest.mark.parametrize(
    ("operand_shape", "expected"),
    [
        pytest.param([2, 3], True, id="static-operand"),
        pytest.param([DYNAMIC_INDEX, 3], False, id="dynamic-operand"),
    ],
)
def test_reshape_is_speculatable(operand_shape: list[int], expected: bool):
    op = _create_reshape_op(operand_shape, [6])

    assert op.is_speculatable() is expected
