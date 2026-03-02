"""
Traits specific to the StableHLO dialect.
"""

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation
from xdsl.traits import ConditionallySpeculatable, RecursivelySpeculatable


class RecursivelySpeculatableIfAllInputsStatic(ConditionallySpeculatable):
    @classmethod
    def is_speculatable(cls, op: Operation):
        inputs_static = all(
            isinstance(operand_type, TensorType) and operand_type.has_static_shape()
            for operand_type in op.operand_types
        )
        return inputs_static and RecursivelySpeculatable.is_speculatable(op)
