"""Tests for StableHLO dynamism operations."""

import pytest
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    TensorType,
    i32,
    i64,
)
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.dynamism_ops import DynamicBroadcastInDimOp
from xdsl_jax.dialects.stablehlo.miscellaneous_ops import ConstantOp

_BROADCAST_DIMENSIONS = DenseArrayBase.from_list(i64, [0])


def _create_dynamic_broadcast_in_dim_op(
    *,
    operand_shape: list[int],
    output_dimensions_is_constant: bool,
    result_shape: list[int],
) -> DynamicBroadcastInDimOp:
    result_rank = len(result_shape)
    output_dimensions_type = TensorType(i64, [result_rank])
    output_dimensions = (
        ConstantOp(
            DenseIntOrFPElementsAttr.from_list(output_dimensions_type, [result_rank])
        ).output
        if output_dimensions_is_constant
        else create_ssa_value(output_dimensions_type)
    )
    return DynamicBroadcastInDimOp.create(
        operands=[
            create_ssa_value(TensorType(i32, operand_shape)),
            output_dimensions,
        ],
        result_types=[TensorType(i32, result_shape)],
        properties={"broadcast_dimensions": _BROADCAST_DIMENSIONS},
    )


@pytest.mark.parametrize(
    (
        "operand_shape",
        "output_dimensions_is_constant",
        "result_shape",
        "expected",
    ),
    [
        pytest.param(
            [DYNAMIC_INDEX],
            False,
            [DYNAMIC_INDEX],
            False,
            id="dynamic-input",
        ),
        pytest.param(
            [1],
            False,
            [DYNAMIC_INDEX],
            True,
            id="scalar-to-fully-dynamic-result",
        ),
        pytest.param(
            [2],
            True,
            [2],
            True,
            id="constant-output-dimensions",
        ),
        pytest.param(
            [2],
            False,
            [2],
            False,
            id="non-constant-output-dimensions",
        ),
    ],
)
def test_dynamic_broadcast_in_dim_is_speculatable(
    operand_shape: list[int],
    output_dimensions_is_constant: bool,
    result_shape: list[int],
    expected: bool,
):
    op = _create_dynamic_broadcast_in_dim_op(
        operand_shape=operand_shape,
        output_dimensions_is_constant=output_dimensions_is_constant,
        result_shape=result_shape,
    )

    assert op.is_speculatable() is expected
