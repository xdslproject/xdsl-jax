"""
Data movement operations for the StableHLO dialect.
"""

from xdsl.dialects.builtin import AnyTensorType, DenseArrayBase, IntegerAttr, i64
from xdsl.irdl import (
    AtLeast,
    IRDLOperation,
    eq,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    ConditionallySpeculatable,
    NoMemoryEffect,
    Pure,
)
from xdsl.utils.type import get_element_type_or_self

from xdsl_jax.xdsl_extras import (
    AllMatchSameOperatorTrait,
    SameOperandsAndResultElementType,
)

from .custom_directives import SliceRanges, VariadicOperandWithAttribute
from .traits import SpeculatableIfStaticDimInOutputIsStaticInInput
from .types import ScalarIntTensorType


@irdl_op_definition
class ConcatenateOp(IRDLOperation):
    """
    Concatenates a variadic number of tensors in ``inputs`` along ``dimension``
    dimension in the same order as the given arguments and produces a ``result``
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = stablehlo.concatenate %input0, %input1, dim = 0
    : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
    """

    name = "stablehlo.concatenate"

    inputs = var_operand_def(AnyTensorType)
    result = result_def(AnyTensorType)
    dimension = prop_def(IntegerAttr.constr(type=eq(i64), value=AtLeast(0)))

    traits = traits_def(
        NoMemoryEffect(),
        ConditionallySpeculatable(),
        SameOperandsAndResultElementType(),
    )

    assembly_format = (
        "custom<VariadicOperandWithAttribute>($inputs) "
        "`dim` `=` $dimension attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (VariadicOperandWithAttribute,)


@irdl_op_definition
class DynamicSliceOp(IRDLOperation):
    """
    Extracts a slice from the ``operand`` using dynamically-computed starting
    indices and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

    Example:
    ```mlir
    %result = stablehlo.dynamic_slice %operand, %start_indices0, %start_indices1,
      sizes = [2, 2] : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
    ```
    """

    name = "stablehlo.dynamic_slice"
    operand = operand_def(AnyTensorType)
    start_indices = var_operand_def(ScalarIntTensorType)
    slice_sizes = prop_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)

    traits = traits_def(
        Pure(),
        AllMatchSameOperatorTrait(
            ("operand", "result"),
            lambda x: get_element_type_or_self(x.type),
            "element type",
        ),
    )

    assembly_format = (
        "$operand `,` custom<VariadicOperandWithAttribute>($start_indices) "
        "`sizes` `=` $slice_sizes attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (VariadicOperandWithAttribute,)


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
