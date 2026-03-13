"""Tests for StableHLO data movement operations."""

import pytest
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    I64,
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    IntegerAttr,
    TensorType,
    i32,
    i64,
)
from xdsl.ir import Block, Region
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.attributes import (
    GatherDimensionNumbers,
    ScatterDimensionNumbers,
)
from xdsl_jax.dialects.stablehlo.data_movement_ops import GatherOp, ScatterOp
from xdsl_jax.dialects.stablehlo.ops import ReturnOp

_EMPTY_DIMS = ArrayAttr[IntegerAttr[I64]](())
_ZERO = IntegerAttr(0, i64)

_GATHER_DIMENSION_NUMBERS = GatherDimensionNumbers(
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _ZERO,
)

_SCATTER_DIMENSION_NUMBERS = ScatterDimensionNumbers(
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _EMPTY_DIMS,
    _ZERO,
)


def _create_gather_op(
    *, operand_shape: list[int], indices_are_sorted: bool
) -> GatherOp:
    return GatherOp.create(
        operands=[
            create_ssa_value(TensorType(i32, operand_shape)),
            create_ssa_value(TensorType(i64, [1])),
        ],
        result_types=[TensorType(i32, [1])],
        properties={
            "dimension_numbers": _GATHER_DIMENSION_NUMBERS,
            "slice_sizes": DenseArrayBase.from_list(i64, [1]),
            "indices_are_sorted": BoolAttr.from_bool(indices_are_sorted),
        },
    )


def _create_scatter_op(
    *,
    input_shape: list[int],
    indices_are_sorted: bool,
    unique_indices: bool,
) -> ScatterOp:
    block = Block(arg_types=[TensorType(i32, []), TensorType(i32, [])])
    block.add_op(ReturnOp([block.args[0]]))
    return ScatterOp.create(
        operands=[
            create_ssa_value(TensorType(i32, input_shape)),
            create_ssa_value(TensorType(i64, [1])),
            create_ssa_value(TensorType(i32, [1])),
        ],
        result_types=[TensorType(i32, input_shape)],
        properties={
            "scatter_dimension_numbers": _SCATTER_DIMENSION_NUMBERS,
            "indices_are_sorted": BoolAttr.from_bool(indices_are_sorted),
            "unique_indices": BoolAttr.from_bool(unique_indices),
        },
        regions=[Region(block)],
    )


@pytest.mark.parametrize(
    ("indices_are_sorted", "expected"),
    [(False, True), (True, False)],
)
def test_gather_is_speculatable(indices_are_sorted: bool, expected: bool):
    op = _create_gather_op(operand_shape=[2], indices_are_sorted=indices_are_sorted)

    assert op.is_speculatable() is expected


@pytest.mark.parametrize(
    ("input_shape", "indices_are_sorted", "unique_indices", "expected"),
    [
        ([2], False, True, False),
        ([DYNAMIC_INDEX], False, False, False),
        ([2], False, False, True),
    ],
)
def test_scatter_is_speculatable(
    input_shape: list[int],
    indices_are_sorted: bool,
    unique_indices: bool,
    expected: bool,
):
    op = _create_scatter_op(
        input_shape=input_shape,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )

    assert op.is_speculatable() is expected
