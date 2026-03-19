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
from xdsl_jax.dialects.stablehlo.data_movement_ops import (
    ConcatenateOp,
    GatherOp,
    PadOp,
    ReshapeOp,
    ScatterOp,
    TransposeOp,
)
from xdsl_jax.dialects.stablehlo.modularity_ops import ReturnOp

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


def _create_reshape_op(operand_shape: list[int], result_shape: list[int]) -> ReshapeOp:
    operand = create_ssa_value(TensorType(i32, operand_shape))
    return ReshapeOp.create(
        operands=[operand],
        result_types=[TensorType(i32, result_shape)],
    )


def test_pad_constructor():
    op = PadOp(
        operand=create_ssa_value(TensorType(i32, [2])),
        padding_value=create_ssa_value(TensorType(i32, [])),
        edge_padding_low=DenseArrayBase.from_list(i64, [1]),
        edge_padding_high=DenseArrayBase.from_list(i64, [1]),
        interior_padding=DenseArrayBase.from_list(i64, [0]),
        result_type=TensorType(i32, [4]),
    )
    assert op.get_edge_padding_low() == (1,)
    assert op.get_edge_padding_high() == (1,)
    assert op.get_interior_padding() == (0,)


def test_transpose_constructor():
    op = TransposeOp(
        operand=create_ssa_value(TensorType(i32, [2, 3])),
        permutation=DenseArrayBase.from_list(i64, [1, 0]),
        result_type=TensorType(i32, [3, 2]),
    )
    assert op.get_permutation() == (1, 0)


@pytest.mark.parametrize(
    ("indices_are_sorted", "expected"),
    [
        pytest.param(False, True, id="unsorted-indices"),
        pytest.param(True, False, id="sorted-indices"),
    ],
)
def test_gather_is_speculatable(indices_are_sorted: bool, expected: bool):
    op = _create_gather_op(operand_shape=[2], indices_are_sorted=indices_are_sorted)
    assert op.is_speculatable() is expected


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


@pytest.mark.parametrize(
    ("input_shape", "indices_are_sorted", "unique_indices", "expected"),
    [
        pytest.param([2], False, True, False, id="unique-indices"),
        pytest.param(
            [DYNAMIC_INDEX],
            False,
            False,
            False,
            id="dynamic-input-shape",
        ),
        pytest.param([2], False, False, True, id="static-inputs"),
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
