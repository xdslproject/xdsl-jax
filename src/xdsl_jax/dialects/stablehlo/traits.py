"""
Traits specific to the StableHLO dialect.
"""

from typing import cast

from xdsl.dialects.builtin import DYNAMIC_INDEX, TensorType
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


class RecursivelySpeculatableIfStaticDimInOutputIsStaticInInput(
    ConditionallySpeculatable
):
    @classmethod
    def is_speculatable(cls, op: Operation):
        if not op.operands or not op.results:
            return False
        input_type = cast(TensorType, op.operand_types[0])
        result_type = cast(TensorType, op.result_types[0])

        input_shape = input_type.get_shape()
        result_shape = result_type.get_shape()
        if len(input_shape) != len(result_shape):
            return False

        for idx, result_dim in enumerate(result_shape):
            if result_dim != DYNAMIC_INDEX and input_shape[idx] == DYNAMIC_INDEX:
                return False
        return RecursivelySpeculatable.is_speculatable(op)
