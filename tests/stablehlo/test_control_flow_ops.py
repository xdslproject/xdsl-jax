"""Tests for StableHLO control flow operations."""

from xdsl.dialects.builtin import TensorType, i32
from xdsl.ir import Block, Region
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.attributes import TokenType
from xdsl_jax.dialects.stablehlo.control_flow_ops import AfterAllOp, CaseOp
from xdsl_jax.dialects.stablehlo.modularity_ops import ReturnOp

_TENSOR_I32_2 = TensorType(i32, [2])


def test_after_all_constructor():
    token = create_ssa_value(TokenType())
    op = AfterAllOp([token])
    assert op.result.type == TokenType()


def test_case_constructor():
    op = CaseOp(
        index=create_ssa_value(_TENSOR_I32_2),
        branches=[Region(Block()), Region(Block())],
        result_types=[_TENSOR_I32_2],
    )
    assert op.results[0].type == _TENSOR_I32_2


def test_return_constructor():
    op = ReturnOp(
        [
            create_ssa_value(_TENSOR_I32_2),
        ]
    )
    assert len(op.input) == 1
    assert op.input[0].type == _TENSOR_I32_2
