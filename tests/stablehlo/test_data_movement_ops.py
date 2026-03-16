"""Tests for StableHLO data movement operations."""

import pytest
from xdsl.dialects.builtin import DYNAMIC_INDEX, IntegerAttr, TensorType, i32, i64
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.data_movement_ops import ConcatenateOp, ReshapeOp


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


def _create_concatenate_op(
    operand_shapes: list[list[int]], result_shape: list[int]
) -> ConcatenateOp:
    operands = [create_ssa_value(TensorType(i32, shape)) for shape in operand_shapes]
    return ConcatenateOp.create(
        operands=operands,
        result_types=[TensorType(i32, result_shape)],
        properties={"dimension": IntegerAttr(0, i64)},
    )


def test_concatenate_is_not_speculatable_without_operands_or_results():
    op = ConcatenateOp.create(properties={"dimension": IntegerAttr(0, i64)})

    assert not op.is_speculatable()


@pytest.mark.parametrize(
    ("operand_shapes", "result_shape", "expected"),
    [
        pytest.param([[3, 2], [1, 2]], [4, 2], True, id="all-static"),
        pytest.param(
            [[DYNAMIC_INDEX, 2], [DYNAMIC_INDEX, 2]],
            [DYNAMIC_INDEX, 2],
            True,
            id="dynamic-concat-dimension",
        ),
        pytest.param(
            [[3, DYNAMIC_INDEX], [1, 2]],
            [4, 2],
            False,
            id="dynamic-non-concat-dimension",
        ),
    ],
)
def test_concatenate_is_speculatable(
    operand_shapes: list[list[int]], result_shape: list[int], expected: bool
):
    op = _create_concatenate_op(operand_shapes, result_shape)

    assert op.is_speculatable() is expected
