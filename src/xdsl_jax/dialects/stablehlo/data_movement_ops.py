"""
Data movement operations for the StableHLO dialect.
"""

from xdsl.dialects.builtin import AnyTensorType, DenseArrayBase, TensorType, i64
from xdsl.ir import Operation, cast
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import (
    ConditionallySpeculatable,
    NoMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException

from xdsl_jax.xdsl_extras import (
    AllMatchSameOperatorTrait,
    SameOperandsAndResultElementType,
)

from .custom_directives import SliceRanges
from .traits import SpeculatableIfStaticDimInOutputIsStaticInInput


@irdl_op_definition
class ReshapeOp(IRDLOperation):
    """
    Performs reshape of ``operand`` tensor to a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %result = stablehlo.reshape %operand : (tensor<2xf32>) -> tensor<1x2xf32>
    ```
    """

    name = "stablehlo.reshape"
    operand = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    assembly_format = """
    operands attr-dict `:` functional-type(operands, results)
    """

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultElementType(),
    )

    @classmethod
    def is_speculatable(cls, op: Operation) -> bool:
        operand_type = op.operand_types[0]
        return isinstance(operand_type, TensorType) and operand_type.has_static_shape()

    def verify_(self) -> None:
        """Verify that the operation has the same shape for all operands and results."""
        o_type = cast(TensorType, self.operand_types[0])
        r_type = cast(TensorType, self.result_types[0])

        # If o_type or r_type is dynamically shaped there is nothing to verify.
        if not o_type.has_static_shape() or not r_type.has_static_shape():
            return

        # If the operand type is statically shaped (not required) the number of
        # elements must match that of the result type.
        num_operand_elements = 1
        for dim in o_type.get_shape():
            num_operand_elements *= dim

        num_result_elements = 1
        for dim in r_type.get_shape():
            num_result_elements *= dim

        if num_result_elements != num_operand_elements:
            raise VerifyException(
                "number of output elements ("
                f"{num_result_elements}"
                ") doesn't match expected number of elements ("
                f"{num_operand_elements}"
                ")"
            )


@irdl_op_definition
class SliceOp(IRDLOperation):
    """
    Extracts a slice from the ``operand`` using statically-computed starting
    indices and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = stablehlo.slice %operand [1:3, 4:8:2]
       : (tensor<3x8xi64>) -> tensor<2x2xi64>

    // Same in generic form: the `1:3` above is mapped to the first entry in
    // `start_indices` and `limit_indices`, while `strides` is implicitly 1.
    // The `4:8:2` above is parsed into the second entry of `start_indices`,
    // `limit_indices` and `strides` respectively.
    %result = "stablehlo.slice" (%operand) {
      start_indices = array<i64: 1, 4>,
      limit_indices = array<i64: 3, 8>,
      strides = array<i64: 1, 2>
    } : (tensor<3x8xi64>) -> tensor<2x2xi64>
    ```
    """

    name = "stablehlo.slice"

    operand = operand_def(AnyTensorType)
    start_indices = prop_def(DenseArrayBase.constr(i64))
    limit_indices = prop_def(DenseArrayBase.constr(i64))
    strides = prop_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)

    assembly_format = (
        "$operand custom<SliceRanges>($start_indices, $limit_indices, $strides)"
        "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (SliceRanges,)

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        SpeculatableIfStaticDimInOutputIsStaticInInput(),
        AllMatchSameOperatorTrait(
            ("start_indices", "limit_indices", "strides"), len, "size"
        ),
        SameOperandsAndResultElementType(),
    )
