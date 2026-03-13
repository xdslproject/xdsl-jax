"""Tests for StableHLO data movement operations."""

from xdsl.dialects.builtin import DYNAMIC_INDEX, IntegerAttr, TensorType, i32, i64
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.data_movement_ops import ConcatenateOp


def _create_concatenate_op(
    operand_shapes: list[list[int]], result_shape: list[int], *, dimension: int
) -> ConcatenateOp:
    operands = [create_ssa_value(TensorType(i32, shape)) for shape in operand_shapes]
    return ConcatenateOp.create(
        operands=operands,
        result_types=[TensorType(i32, result_shape)],
        properties={"dimension": IntegerAttr(dimension, i64)},
    )


def test_concatenate_is_not_speculatable_without_operands_or_results():
    op = ConcatenateOp.create(properties={"dimension": IntegerAttr(0, i64)})

    assert not op.is_speculatable()


def test_concatenate_is_speculatable_with_static_shapes():
    op = _create_concatenate_op([[3, 2], [1, 2]], [4, 2], dimension=0)

    assert op.is_speculatable()


def test_concatenate_is_speculatable_with_dynamic_concat_dimension():
    op = _create_concatenate_op(
        [[DYNAMIC_INDEX, 2], [DYNAMIC_INDEX, 2]],
        [DYNAMIC_INDEX, 2],
        dimension=0,
    )

    assert op.is_speculatable()


def test_concatenate_is_not_speculatable_with_dynamic_non_concat_dimension():
    op = _create_concatenate_op([[3, DYNAMIC_INDEX], [1, 2]], [4, 2], dimension=0)

    assert not op.is_speculatable()
