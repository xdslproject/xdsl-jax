"""Constructor tests for StableHLO elementwise operations."""

from xdsl.dialects.builtin import TensorType, i32
from xdsl.utils.test_value import create_ssa_value

from xdsl_jax.dialects.stablehlo.elementwise_binary_ops import AddOp
from xdsl_jax.dialects.stablehlo.elementwise_other_ops import BitcastConvertOp
from xdsl_jax.dialects.stablehlo.elementwise_unary_ops import AbsOp

_TENSOR_I32_2 = TensorType(i32, [2])


def test_add_constructor():
    lhs = create_ssa_value(_TENSOR_I32_2)
    rhs = create_ssa_value(_TENSOR_I32_2)
    op = AddOp(lhs, rhs)
    assert op.result.type == lhs.type


def test_abs_constructor():
    operand = create_ssa_value(_TENSOR_I32_2)
    op = AbsOp(operand)
    assert op.result.type == operand.type


def test_bitcast_convert_constructor():
    operand = create_ssa_value(_TENSOR_I32_2)
    op = BitcastConvertOp(operand, _TENSOR_I32_2)
    assert op.result.type == _TENSOR_I32_2
