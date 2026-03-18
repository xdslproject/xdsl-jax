"""Tests for StableHLO miscellaneous operations."""

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, i32

from xdsl_jax.dialects.stablehlo.miscellaneous_ops import ConstantOp

_TENSOR_I32_1 = TensorType(i32, [1])


def test_constant_constructor():
    value = DenseIntOrFPElementsAttr.from_list(_TENSOR_I32_1, [7])
    op = ConstantOp(value)
    assert op.output.type == _TENSOR_I32_1
    assert op.value == value
