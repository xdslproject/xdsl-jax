"""
Data movement operations for the StableHLO dialect.
"""

from xdsl.dialects.builtin import AnyTensorType, DenseArrayBase, i64
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

from xdsl_jax.xdsl_extras import (
    AllMatchSameOperatorTrait,
    SameOperandsAndResultElementType,
)

from .custom_directives import SliceRanges
from .traits import SpeculatableIfStaticDimInOutputIsStaticInInput


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
