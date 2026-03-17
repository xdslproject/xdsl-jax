"""
[StableHLO](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
is an operation set for high-level operations (HLO) in machine learning (ML) models.
StableHLO works as a portability layer between different ML frameworks and ML compilers:
ML frameworks that produce StableHLO programs are compatible with ML compilers that
consume StableHLO programs.
"""

from collections.abc import Sequence
from typing import cast

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnyTensorType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    TensorType,
    i32,
    i64,
)
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.irdl.attributes import eq
from xdsl.irdl.constraints import AtLeast
from xdsl.traits import (
    ConditionallySpeculatable,
    ConstantLike,
    IsTerminator,
    NoMemoryEffect,
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.type import have_compatible_shape

from xdsl_jax.xdsl_extras.traits import (
    Elementwise,
    SameOperandsAndResultElementType,
    SameOperandsAndResultShape,
)

from .attributes import ComparisonDirectionAttr, ComparisonTypeAttr, TokenType
from .custom_directives import (
    ConstantOpValue,
    ExponentMantissa,
    SameOperandsAndResultType,
    SelectOpType,
)
from .traits import (
    CompatibleOperandsAndResultType,
    SpeculatableIfAllInputsStatic,
    SpeculatableIfStaticDimInOutputIsStaticInInput,
    have_compatible_type_sequences,
)
from .types import (
    FloatTensorType,
    IntOrFloatOrComplexTensorType,
    PredTensorType,
    SI32TensorType,
    TensorOrTokenOrBufferType,
    TensorOrTokenType,
)


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """This op is un-documented.

    StableHLO's return is used inside of the bodies of StableHLO ops.
    It behaves like func.return but for StableHLO ops.
    The func.return op is used inside of func.func op.

    https://discord.com/channels/999073994483433573/1259494021269688360/1259992088565645312
    """

    name = "stablehlo.return"

    input = var_operand_def(TensorOrTokenOrBufferType)

    traits = traits_def(Pure(), IsTerminator())

    assembly_format = "$input attr-dict (`:` type($input)^)?"

    def __init__(self, input: list[SSAValue]):
        super().__init__(operands=(input,))


@irdl_op_definition
class AfterAllOp(IRDLOperation):
    """
    Ensures that the operations producing the inputs are executed before any operations
    that depend on result.
    Execution of this operation does nothing, it only exists to establish data
    dependencies from result to inputs.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all)
    """

    name = "stablehlo.after_all"
    inputs = var_operand_def(TokenType)
    result = result_def(TokenType)

    traits = traits_def(Pure())

    assembly_format = (
        "$inputs attr-dict"
        " `:` custom<SameOperandsAndResultType>"
        "(type($inputs), type($result))"
    )
    custom_directives = (SameOperandsAndResultType,)

    def __init__(self, inputs: Sequence[SSAValue]):
        super().__init__(operands=[inputs], result_types=(TokenType(),))


@irdl_op_definition
class BitcastConvertOp(IRDLOperation):
    """
    Performs a bitcast operation on operand tensor and produces a result tensor
    where the bits of the entire operand tensor are reinterpreted using the type of the
    result tensor.

    More formally, given `E = element_type(operand)`, `E' = element_type(result)`,
    and `R = rank(operand)`:

    * If `num_bits(E') < num_bits(E)`, `bits(result[i0, ..., iR-1, :]) = bits(operand[i0, ..., iR-1])`.
    * If `num_bits(E') > num_bits(E)`, `bits(result[i0, ..., iR-2]) = bits(operand[i0, ..., iR-2, :])`.
    * If `num_bits(E') = num_bits(E)`, `bits(result[i0, ..., iR-1]) = bits(operand[i0, ..., iR-1])`.

    `bits` returns in-memory representation of a given value,
    and its behavior is implementation-defined because the exact representation of
    tensors is implementation-defined,
    and the exact representation of element types is implementation-defined as well.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert)
    """  # noqa: E501

    name = "stablehlo.bitcast_convert"
    input = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    traits = traits_def(NoMemoryEffect(), ConditionallySpeculatable())

    def __init__(self, input: SSAValue, result: Attribute):
        super().__init__(operands=(input,), result_types=(result,))


@irdl_op_definition
class CaseOp(IRDLOperation):
    """
    Produces the output from executing exactly one function from `branches`
    depending on the value of `index`. More formally, `result = selected_branch()`
    where:

    * `selected_branch = branches[index]` if `0 <= index < size(branches)`.
    * `selected_branch = branches[-1]` otherwise.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case)
    """

    name = "stablehlo.case"
    index = operand_def(SI32TensorType)
    branches = var_region_def("single_block")
    _results = var_result_def(TensorOrTokenType)

    traits = traits_def(
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
        SingleBlockImplicitTerminator(ReturnOp),
    )

    def __init__(
        self,
        index: SSAValue,
        branches: Sequence[Region],
        result_types: Sequence[AnyTensorType | TokenType],
    ):
        super().__init__(
            operands=(index,), result_types=(result_types,), regions=(branches,)
        )


@irdl_op_definition
class ClampOp(IRDLOperation):
    """Element-wise clamp with min and max bounds.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp
    """

    name = "stablehlo.clamp"

    min = operand_def(AnyTensorType)
    operand = operand_def(AnyTensorType)
    max = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    assembly_format = (
        "$min `,` $operand `,` $max attr-dict `:` "
        "custom<SameOperandsAndResultType>(type(operands), type(results))"
    )

    custom_directives = (SameOperandsAndResultType,)

    traits = traits_def(
        NoMemoryEffect(),
    )


@irdl_op_definition
class CompareOp(IRDLOperation):
    """Element-wise compare with direction and type attributes."""

    name = "stablehlo.compare"

    assembly_format = (
        "$comparison_direction `,` $lhs `,` $rhs (`,` $comparison_type^)? "
        "attr-dict `:` functional-type(operands, results)"
    )

    lhs = operand_def(AnyTensorType)
    rhs = operand_def(AnyTensorType)
    result = result_def(PredTensorType)
    comparison_direction = attr_def(ComparisonDirectionAttr)
    comparison_type = opt_prop_def(ComparisonTypeAttr)

    traits = traits_def(
        NoMemoryEffect(),
        Elementwise(),
        SameOperandsAndResultShape(),
    )


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """
    Produces an `output` tensor from a constant `value`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant)
    """

    name = "stablehlo.constant"

    value = attr_def(DenseIntOrFPElementsAttr)
    output = result_def(AnyTensorType)

    traits = traits_def(Pure(), ConstantLike())

    assembly_format = "attr-dict custom <ConstantOpValue>($value, type($output))"
    custom_directives = (ConstantOpValue,)

    def __init__(self, value: DenseIntOrFPElementsAttr):
        super().__init__(attributes={"value": value}, result_types=(value.type,))


@irdl_op_definition
class IotaOp(IRDLOperation):
    """
    Fills an `output` tensor with values in increasing order starting from zero
    along the `iota_dimension` dimension.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

    Example:
    ```mlir
    %output = stablehlo.iota dim = 0 : tensor<4x5xi32>
    """

    name = "stablehlo.iota"

    iota_dimension = prop_def(IntegerAttr.constr(type=eq(i64), value=AtLeast(0)))

    output = result_def(IntOrFloatOrComplexTensorType)

    traits = traits_def(Pure())

    assembly_format = "`dim` `=` $iota_dimension attr-dict `:` type($output)"

    def verify_(self) -> None:
        output_type = cast(TensorType[Attribute], self.output.type)
        if not output_type.has_static_shape():
            raise VerifyException("Iota output must have a static shape.")

        rank = len(output_type.get_shape())
        if rank == 0:
            raise VerifyException("Iota does not support scalars.")
        if self.iota_dimension.value.data >= rank:
            raise VerifyException("Iota dimension cannot go beyond the output rank.")


@irdl_op_definition
class MapOp(IRDLOperation):
    """
    Applies a map function `computation` to `inputs` along the `dimensions` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#map

    Example:
    ```mlir
    %result = "stablehlo.map"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 0, 1>
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
    ```
    """

    name = "stablehlo.map"

    inputs = var_operand_def(AnyTensorType)
    dimensions = attr_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)
    computation = var_region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        SameOperandsAndResultShape(),
        SingleBlockImplicitTerminator(ReturnOp),
    )

    def _verify_computation_block(self, block: Block) -> None:
        """Verify that computation block has the correct arguments and return type.
        it checks the following:
        - The number of operands and arguments match.
        - The arguments are 0-rank tensors
        - The element types of the arguments match the operand element types.
        - The return value is a single 0-rank tensor.
        """
        block_args = block.args
        if len(self.inputs) != len(block_args):
            raise VerifyException(
                "expects number of operands to match the arity of map computation, "
                f"but got: {len(self.inputs)} and {len(block_args)}"
            )

        for idx, arg in enumerate(block_args):
            arg_type = cast(TensorType[Attribute], arg.type)
            if arg_type.get_num_dims() != 0:
                raise VerifyException(
                    "computation arguments must be 0-rank tensor, but got: "
                    f"arg #{idx} of type {arg.type}"
                )
            operand_elem_type = cast(
                TensorType[Attribute], self.inputs[idx].type
            ).element_type
            if arg_type.element_type != operand_elem_type:
                raise VerifyException(
                    "element type of operands and computation arguments must match, "
                    f"but got: {operand_elem_type} and {arg_type.element_type}"
                )

        terminator = cast(ReturnOp, block.last_op)
        if len(terminator.input) != 1:
            raise VerifyException(
                "computation must return single output, "
                f"but got: {len(terminator.input)}"
            )

        computation_output_type = cast(TensorType[Attribute], terminator.input[0].type)
        if computation_output_type.get_num_dims() != 0:
            raise VerifyException(
                "computation must return 0-rank tensor, but got: "
                f"{terminator.input[0].type}"
            )

    def _verify_dimensions(self, dimensions: tuple[int, ...]) -> None:
        """Verify the dimensions are monotonically increasing and
        the operand dimensions are a subset of the map dimensions."""
        for idx, dim in enumerate(dimensions):
            if dim != idx:
                raise VerifyException(
                    "requires monotonically increasing dimension numbers, "
                    f"but got: {dimensions}"
                )
        for operand in self.inputs:
            operand_type = cast(TensorType[Attribute], operand.type)
            if len(dimensions) != len(operand_type.get_shape()):
                raise VerifyException(
                    "applied to a subset of dimensions currently not supported: "
                    f"operand dimensions = {len(operand_type.get_shape())}, "
                    f"requested map dimensions size = {len(dimensions)}"
                )

    def verify_(self) -> None:
        computation = self.computation[0]
        dimensions = self.dimensions.get_values()
        self._verify_computation_block(computation.block)
        self._verify_dimensions(dimensions)


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Permutes the dimensions of `operand` tensor using `permutation` and produces a
    `result` tensor. More formally, `result[result_index] = operand[operand_index]`
    where `result_index[d] = operand_index[permutation[d]]`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose)
    """

    name = "stablehlo.transpose"

    operand = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)
    permutation = attr_def(DenseArrayBase.constr(i64))

    traits = traits_def(NoMemoryEffect(), ConditionallySpeculatable())

    assembly_format = (
        "$operand `,` `dims` `=` $permutation "
        "attr-dict `:` functional-type(operands, results)"
    )

    def __init__(
        self, operand: SSAValue, permutation: DenseArrayBase, result_type: Attribute
    ):
        super().__init__(
            operands=(operand,),
            result_types=(result_type,),
            attributes={"permutation": permutation},
        )

    def get_permutation(self) -> tuple[int, ...]:
        return self.permutation.get_values()

    def verify_(self) -> None:
        # Operand and result types are checked before the custom `verify_`
        o_type = cast(TensorType[Attribute], self.operand.type)
        r_type = self.result.type

        o_shape = o_type.get_shape()
        r_shape = r_type.get_shape()

        # TODO: Quantization constraints
        # `permutation` is a permutation of `range(rank(operand))`
        permutation = self.get_permutation()
        if sorted(permutation) != list(range(len(o_shape))):
            raise VerifyException(
                f"Permutation {permutation} of transpose must be a permutation of "
                f"range({len(o_shape)})"
            )

        # `shape(result) = dim(operand, permutation...)`
        for i, dim in enumerate(permutation):
            if r_shape[i] != o_shape[dim]:
                raise VerifyException(
                    f"Permutation mismatch at dimension {i}, expected {o_shape[dim]}"
                )


@irdl_op_definition
class PadOp(IRDLOperation):
    """
    Expands operand by padding around the tensor as well as between the
    elements of the tensor with the given padding_value.

    edge_padding_low and edge_padding_high specify the amount of padding
    added at the low-end (next to index 0) and the high-end
    (next to the highest index) of each dimension respectively.
    The amount of padding can be negative, where the absolute value of negative
    padding indicates the number of elements to remove from the specified dimension.

    interior_padding specifies the amount of padding added between any
    two elements in each dimension which may not be negative. Interior padding occurs
    before edge padding such that negative edge padding will remove elements from
    the interior-padded operand.

    More formally, result[result_index] is defined as:

    operand[operand_index] if
    result_index = edge_padding_low + operand_index * (interior_padding + 1).
    padding_value otherwise.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad)
    """

    name = "stablehlo.pad"

    operand = operand_def(AnyTensorType)
    padding_value = operand_def(SI32TensorType)
    result = result_def(AnyTensorType)
    edge_padding_low = attr_def(DenseArrayBase.constr(i64))
    edge_padding_high = attr_def(DenseArrayBase.constr(i64))
    interior_padding = attr_def(DenseArrayBase.constr(i64))

    traits = traits_def(NoMemoryEffect(), SameOperandsAndResultElementType())

    assembly_format = (
        "$operand `,` $padding_value `,` "
        "`low` `=` $edge_padding_low `,` "
        "`high` `=` $edge_padding_high `,` "
        "`interior` `=` $interior_padding "
        "attr-dict `:` functional-type(operands, results)"
    )

    def __init__(
        self,
        operand: SSAValue,
        padding_value: SSAValue,
        edge_padding_low: DenseArrayBase,
        edge_padding_high: DenseArrayBase,
        interior_padding: DenseArrayBase,
        result_type: Attribute,
    ):
        super().__init__(
            operands=(operand, padding_value),
            result_types=(result_type,),
            attributes={
                "edge_padding_low": edge_padding_low,
                "edge_padding_high": edge_padding_high,
                "interior_padding": interior_padding,
            },
        )

    def get_edge_padding_low(self) -> tuple[int, ...]:
        return self.edge_padding_low.get_values()

    def get_edge_padding_high(self) -> tuple[int, ...]:
        return self.edge_padding_high.get_values()

    def get_interior_padding(self) -> tuple[int, ...]:
        return self.interior_padding.get_values()

    def verify_(self) -> None:
        # Operand and result types are checked before the custom `verify_`
        o_type = cast(TensorType[Attribute], self.operand.type)
        pad_val_type = cast(TensorType[Attribute], self.padding_value.type)
        r_type = self.result.type

        o_shape = o_type.get_shape()
        pad_val_shape = pad_val_type.get_shape()
        r_shape = r_type.get_shape()

        if pad_val_shape:
            raise VerifyException(
                f"Expect padding_value is an 0-dimensional tensor,"
                f" found {pad_val_shape}"
            )

        o_rank = len(o_shape)
        edge_padding_low = self.get_edge_padding_low()
        edge_padding_high = self.get_edge_padding_high()
        interior_padding = self.get_interior_padding()

        hints = [
            "result shape",
            "edge_padding_low",
            "edge_padding_high",
            "interior_padding",
        ]

        # size(edge_padding_low) = size(edge_padding_high)
        # = size(interior_padding) = rank(operand)
        for shape, hint in zip(
            [r_shape, edge_padding_low, edge_padding_high, interior_padding], hints
        ):
            if o_rank != len(shape):
                raise VerifyException(
                    f"Pad operati"
                    f"on rank mismatch "
                    f"while the operand has {o_rank} dimension(s) and "
                    f"{hint} has {len(shape)} dimension(s)"
                )

        # 0 <= interior_padding
        for inner_padding in interior_padding:
            if inner_padding < 0:
                raise VerifyException(
                    f"The interior_padding value must be equal or larger than 0,"
                    f" found {inner_padding} "
                )

        # shape(result) = shape(operand) + edge_padding_low +
        # max(shape(operand) - 1, 0) * interior_padding + edge_padding_high
        for ith_dim, (r_dim, o_dim, pad_low, pad_high, inner_pad) in enumerate(
            zip(r_shape, o_shape, edge_padding_low, edge_padding_high, interior_padding)
        ):
            if r_dim != o_dim + pad_low + max(o_dim - 1, 0) * inner_pad + pad_high:
                raise VerifyException(
                    f"Pad operation at {ith_dim} dimension  mismatch, while "
                    f"the dimension before {o_dim} and the after is {r_dim} "
                    f"with {pad_low}, {pad_high}, {inner_pad} "
                    f"as low, high and inner padding values"
                )


@irdl_op_definition
class SelectOp(IRDLOperation):
    """
    Produces a `result` tensor where each element is selected from `on_true` or
    `on_false` tensor based on the value of the corresponding element of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select

    Example:
    ```mlir
    %result = stablehlo.select %pred, %on_true, %on_false : tensor<2xi1>, tensor<2xi32>
    ```
    """

    name = "stablehlo.select"

    assembly_format = (
        "operands attr-dict `:`  custom<SelectOpType>(type(operands), type(results))"
    )

    pred = operand_def(PredTensorType)
    on_true = operand_def(AnyTensorType)
    on_false = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    custom_directives = (SelectOpType,)

    traits = traits_def(
        NoMemoryEffect(),
        SpeculatableIfAllInputsStatic(),
    )

    def verify_(self) -> None:
        pred_type = cast(TensorType[Attribute], self.pred.type)
        on_true_type = cast(TensorType[Attribute], self.on_true.type)
        on_false_type = cast(TensorType[Attribute], self.on_false.type)

        if not have_compatible_type_sequences((on_true_type,), (on_false_type,)):
            raise VerifyException(
                "requires compatible types for non-predicate operands"
            )

        if pred_type.get_num_dims() != 0 and not have_compatible_shape(
            pred_type, on_true_type
        ):
            raise VerifyException("requires the same shape for all operands")

        inferred_shape = tuple(
            false_dim if true_dim == DYNAMIC_INDEX else true_dim
            for true_dim, false_dim in zip(
                on_true_type.get_shape(), on_false_type.get_shape()
            )
        )
        inferred_result_type = TensorType(
            on_true_type.element_type,
            inferred_shape,
            on_true_type.encoding,
        )
        result_type = self.result.type

        if not have_compatible_type_sequences((inferred_result_type,), (result_type,)):
            raise VerifyException(
                f"'{self.name}' op inferred type(s) '{inferred_result_type}' are "
                f"incompatible with return type(s) of operation '{result_type}'"
            )


@irdl_op_definition
class ReducePrecisionOp(IRDLOperation):
    """
    Performs element-wise conversion of `operand` to another floating-point type
    that uses `exponent_bits` and `mantissa_bits` and back to the original
    floating-point type and produces an `output` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_precision

    Example:
    ```mlir
    %output = stablehlo.reduce_precision %operand, format = e5m10 : tensor<6xf64>
    ```
    """

    name = "stablehlo.reduce_precision"

    assembly_format = (
        "$operand `,` `format` `=` "
        "custom<ExponentMantissa>($exponent_bits, $mantissa_bits)"
        "attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))"
    )

    custom_directives = (ExponentMantissa, SameOperandsAndResultType)

    operand = operand_def(FloatTensorType)
    result = result_def(FloatTensorType)

    exponent_bits = attr_def(IntegerAttr.constr(type=eq(i32), value=AtLeast(1)))
    mantissa_bits = attr_def(IntegerAttr.constr(type=eq(i32), value=AtLeast(0)))

    traits = traits_def(
        NoMemoryEffect(),
        Elementwise(),
        CompatibleOperandsAndResultType(),
        SpeculatableIfStaticDimInOutputIsStaticInInput(),
    )
