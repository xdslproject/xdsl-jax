"""Tests for StableHLO reduction operations."""

import pytest
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    I64,
    ArrayAttr,
    IntegerAttr,
    TensorType,
    i32,
    i64,
)
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.attributes import DotAttr
from xdsl_jax.dialects.stablehlo.reduction_ops import DotGeneralOp

_DOT_DIMENSION_NUMBERS = DotAttr(
    ArrayAttr[IntegerAttr[I64]](()),
    ArrayAttr[IntegerAttr[I64]](()),
    ArrayAttr((IntegerAttr(1, i64),)),
    ArrayAttr((IntegerAttr(0, i64),)),
)


def _create_dot_general_op(
    lhs_shape: list[int], rhs_shape: list[int], result_shape: list[int]
) -> DotGeneralOp:
    return DotGeneralOp.create(
        operands=[
            create_ssa_value(TensorType(i32, lhs_shape)),
            create_ssa_value(TensorType(i32, rhs_shape)),
        ],
        result_types=[TensorType(i32, result_shape)],
        properties={"dot_dimension_numbers": _DOT_DIMENSION_NUMBERS},
    )


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "result_shape", "expected"),
    [
        pytest.param(
            [2, DYNAMIC_INDEX],
            [3, 4],
            [2, 4],
            False,
            id="lhs-dynamic-contracting-dimension",
        ),
        pytest.param(
            [2, 3],
            [DYNAMIC_INDEX, 4],
            [2, 4],
            False,
            id="rhs-dynamic-contracting-dimension",
        ),
        pytest.param(
            [DYNAMIC_INDEX, 3],
            [3, 4],
            [2, 4],
            False,
            id="lhs-dynamic-feeds-static-result-dimension",
        ),
        pytest.param(
            [2, 3],
            [3, DYNAMIC_INDEX],
            [2, 4],
            False,
            id="rhs-dynamic-feeds-static-result-dimension",
        ),
        pytest.param(
            [DYNAMIC_INDEX, 3],
            [3, DYNAMIC_INDEX],
            [DYNAMIC_INDEX, DYNAMIC_INDEX],
            True,
            id="dynamic-dimensions-feed-dynamic-result-dimensions",
        ),
    ],
)
def test_dot_general_is_speculatable(
    lhs_shape: list[int],
    rhs_shape: list[int],
    result_shape: list[int],
    expected: bool,
):
    op = _create_dot_general_op(lhs_shape, rhs_shape, result_shape)

    assert op.is_speculatable() is expected
