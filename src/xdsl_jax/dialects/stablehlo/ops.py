"""
[StableHLO](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
is an operation set for high-level operations (HLO) in machine learning (ML) models.
StableHLO works as a portability layer between different ML frameworks and ML compilers:
ML frameworks that produce StableHLO programs are compatible with ML compilers that
consume StableHLO programs.
"""

from collections.abc import Sequence
from typing import ClassVar, TypeAlias, cast

from xdsl.dialects.builtin import (
    I32,
    AnyTensorType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    TensorType,
    i64,
)
from xdsl.ir import Attribute, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.traits import (
    ConditionallySpeculatable,
    IsTerminator,
    NoMemoryEffect,
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException

from .attributes import TokenType
from .types import TensorOrTokenOrBufferType, TensorOrTokenType

# TODO: Change to SI32 once StableHLO adopts signful integer semantics
# See: https://github.com/openxla/stablehlo/issues/22
# https://github.com/openxla/stablehlo/issues/2489
SI32TensorType: TypeAlias = TensorType[I32]


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
class ConstantOp(IRDLOperation):
    """
    Produces an `output` tensor from a constant `value`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant)
    """

    name = "stablehlo.constant"

    value = attr_def(DenseIntOrFPElementsAttr)
    output = result_def(AnyTensorType)

    def __init__(self, value: DenseIntOrFPElementsAttr):
        super().__init__(attributes={"value": value}, result_types=(value.type,))


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Permutes the dimensions of `operand` tensor using `permutation` and produces a
    `result` tensor. More formally, `result[result_index] = operand[operand_index]`
    where `result_index[d] = operand_index[permutation[d]]`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose)
    """

    name = "stablehlo.transpose"

    ELEMENT_TYPE: ClassVar = VarConstraint("ELEMENT_TYPE", AnyAttr())

    operand = operand_def(TensorType.constr(ELEMENT_TYPE))
    result = result_def(TensorType.constr(ELEMENT_TYPE))
    permutation = attr_def(DenseArrayBase.constr(i64))

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

    ELEMENT_TYPE: ClassVar = VarConstraint("ELEMENT_TYPE", AnyAttr())

    operand = operand_def(TensorType.constr(ELEMENT_TYPE))
    padding_value = operand_def(TensorType.constr(ELEMENT_TYPE))
    result = result_def(TensorType.constr(ELEMENT_TYPE))
    edge_padding_low = attr_def(DenseArrayBase.constr(i64))
    edge_padding_high = attr_def(DenseArrayBase.constr(i64))
    interior_padding = attr_def(DenseArrayBase.constr(i64))

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
