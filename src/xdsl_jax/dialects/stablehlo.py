"""
[StableHLO](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
is an operation set for high-level operations (HLO) in machine learning (ML) models.
StableHLO works as a portability layer between different ML frameworks and ML compilers:
ML frameworks that produce StableHLO programs are compatible with ML compilers that
consume StableHLO programs.
"""

import abc
from collections.abc import Sequence
from typing import ClassVar, TypeAlias, cast

from xdsl.dialects.builtin import (
    I32,
    AnyFloat,
    AnyTensorType,
    AnyTensorTypeConstr,
    ComplexType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IntegerType,
    TensorType,
    i64,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AnyAttr,
    BaseAttr,
    IRDLOperation,
    VarConstraint,
    attr_def,
    base,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

from xdsl_jax.dialects.attributes import (
    ComparisonDirectionAttr,
    ComparisonTypeAttr,
    DotAttr,
    PrecisionAttr,
    TokenType,
)

IntegerTensorType: TypeAlias = TensorType[IntegerType]
FloatOrComplexType: TypeAlias = AnyFloat | ComplexType
FloatOrComplexTensorType: TypeAlias = TensorType[FloatOrComplexType]
FloatTensorType: TypeAlias = TensorType[AnyFloat]

# TODO: Change to SI32 once StableHLO adopts signful integer semantics
# See: https://github.com/openxla/stablehlo/issues/22
# https://github.com/openxla/stablehlo/issues/2489
SI32TensorType: TypeAlias = TensorType[I32]


# region Abstract Base Classes


# TODO: Abstract the constraint
class ElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    # TODO: Remove this constraint for complex types.
    T: ClassVar = VarConstraint("T", base(AnyTensorType))

    lhs = operand_def(T)
    rhs = operand_def(T)

    result = result_def(T)

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


class IntegerTensorLikeElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", base(IntegerTensorType))

    lhs = operand_def(T)
    rhs = operand_def(T)

    result = result_def(T)

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


class IntegerTensorLikeElementwiseUnaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", base(IntegerTensorType))

    operand = operand_def(T)
    result = result_def(T)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


class FloatOrComplexTensorLikeElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", base(FloatOrComplexTensorType))

    lhs = operand_def(T)
    rhs = operand_def(T)

    result = result_def(T)

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


class FloatOrComplexTensorLikeElementwiseUnaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", base(FloatOrComplexTensorType))

    operand = operand_def(T)
    result = result_def(T)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


class FloatTensorLikeElementwiseUnaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", base(FloatTensorType))

    operand = operand_def(T)
    result = result_def(T)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


# endregion


@irdl_op_definition
class AbsOp(IRDLOperation):
    """
    Performs element-wise abs operation on operand tensor and produces a result tensor.
    Depending on the element type, does the following:

    * For signed integers: integer modulus.
    * For floats: abs from IEEE-754.
    * For complex numbers: complex modulus.
    * For quantized types: dequantize_op_quantize(abs, operand, type(result)).

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs)
    """

    name = "stablehlo.abs"

    # TODO: Remove this constraint for complex types.
    T: ClassVar = VarConstraint("T", base(AnyTensorType))

    operand = operand_def(T)
    result = result_def(T)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            # TODO: Constraints for complex types.
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


@irdl_op_definition
class AddOp(ElementwiseBinaryOperation):
    """
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For booleans: logical OR.
    * For integers: integer addition.
    * For floats: `addition` from IEEE-754.
    * For complex numbers: complex addition.
    * For quantized types: `dequantize_op_quantize(add, lhs, rhs, type(result))`.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add)
    """

    name = "stablehlo.add"


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
class AndOp(IntegerTensorLikeElementwiseBinaryOperation):
    """
    Performs element-wise AND of two tensors lhs and rhs and produces a result tensor.
    Depending on the element type, does the following:

    For booleans: logical AND.
    For integers: bitwise AND.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and)
    """

    name = "stablehlo.and"


@irdl_op_definition
class Atan2Op(FloatOrComplexTensorLikeElementwiseBinaryOperation):
    """
    Performs element-wise atan2 operation on `lhs` and `rhs` tensor and produces a
    `result` tensor. Depending on the element type, does the following:

    * For floats: `atan2` from IEEE-754.
    * For complex numbers: complex atan2.
    * For quantized types: `dequantize_op_quantize(atan2, lhs, rhs, type(result))`.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#atan2)
    """

    name = "stablehlo.atan2"


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
    _results = var_result_def(AnyTensorTypeConstr | BaseAttr(TokenType))

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
class CbrtOp(FloatOrComplexTensorLikeElementwiseUnaryOperation):
    """
    Performs element-wise cubic root operation on `operand` tensor and produces a
    `result` tensor. Depending on the element type, does the following:

    * For floats: `rootn(x, 3)` from IEEE-754.
    * For complex numbers: complex cubic root.
    * For quantized types: `dequantize_op_quantize(cbrt, operand, type(result))`

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cbrt)
    """

    name = "stablehlo.cbrt"


@irdl_op_definition
class CeilOp(FloatTensorLikeElementwiseUnaryOperation):
    """
    Performs element-wise ceil of operand tensor and produces a result tensor.
    Implements the roundToIntegralTowardPositive operation from the IEEE-754
    specification.
    For quantized types, performs dequantize_op_quantize(ceil, operand, type(result)).

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#ceil)
    """

    name = "stablehlo.ceil"


@irdl_op_definition
class CountLeadingZerosOp(IntegerTensorLikeElementwiseUnaryOperation):
    """
    Performs element-wise ceil of `operand` tensor and produces a `result` tensor.
    Implements the `roundToIntegralTowardPositive` operation from the IEEE-754
    specification. For quantized types, performs
    `dequantize_op_quantize(ceil, operand, type(result))`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros)
    """

    name = "stablehlo.count_leading_zeros"


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
class MultiplyOp(ElementwiseBinaryOperation):
    """
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For booleans: logical AND.
    * For integers: integer multiplication.
    * For floats: `multiplication` from IEEE-754.
    * For complex numbers: complex multiplication.
    * For quantized types:
      * `dequantize_op_quantize(multiply, lhs, rhs, type(result))`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply)
    """

    name = "stablehlo.multiply"


@irdl_op_definition
class NotOp(IntegerTensorLikeElementwiseUnaryOperation):
    """
    Performs element-wise NOT of tensor `operand` and produces a `result` tensor.
    Depending on the element type, does the following:

    * For booleans: logical NOT.
    * For integers: bitwise NOT.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not)
    """

    name = "stablehlo.not"


@irdl_op_definition
class OrOp(IntegerTensorLikeElementwiseBinaryOperation):
    """
    Performs element-wise OR of two tensors `lhs` and `rhs` and produces a `result`
    tensor. Depending on the element type, does the following:

    * For booleans: logical OR.
    * For integers: bitwise OR.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or)
    """

    name = "stablehlo.or"


@irdl_op_definition
class PopcntOp(IntegerTensorLikeElementwiseUnaryOperation):
    """
    Performs element-wise count of the number of bits set in the `operand` tensor
    and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt)
    """

    name = "stablehlo.popcnt"


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """This op is un-documented.

    StableHLO's return is used inside of the bodies of StableHLO ops.
    It behaves like func.return but for StableHLO ops.
    The func.return op is used inside of func.func op.

    https://discord.com/channels/999073994483433573/1259494021269688360/1259992088565645312
    """

    name = "stablehlo.return"

    input = var_operand_def(AnyTensorType)
    traits = traits_def(IsTerminator())

    def __init__(self, input: list[SSAValue]):
        super().__init__(operands=(input,))


@irdl_op_definition
class ShiftLeftOp(IntegerTensorLikeElementwiseBinaryOperation):
    """
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs` number
    of bits and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left)
    """

    name = "stablehlo.shift_left"


@irdl_op_definition
class ShiftRightArithmeticOp(IntegerTensorLikeElementwiseBinaryOperation):
    """
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor by
    `rhs` number of bits and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic)
    """

    name = "stablehlo.shift_right_arithmetic"


@irdl_op_definition
class ShiftRightLogicalOp(IntegerTensorLikeElementwiseBinaryOperation):
    """
    Performs element-wise logical right-shift operation on the `lhs` tensor by `rhs`
    number of bits and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical)
    """

    name = "stablehlo.shift_right_logical"


@irdl_op_definition
class SubtractOp(ElementwiseBinaryOperation):
    """
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For integers: integer subtraction.
    * For floats: `subtraction` from IEEE-754.
    * For complex numbers: complex subtraction.
    * For quantized types:
      * `dequantize_op_quantize(subtract, lhs, rhs, type(result))`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract)
    """

    name = "stablehlo.subtract"


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


@irdl_op_definition
class XorOp(IntegerTensorLikeElementwiseBinaryOperation):
    """
    Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a `result`
    tensor. Depending on the element type, does the following:

    * For booleans: logical XOR.
    * For integers: bitwise XOR.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor)
    """

    name = "stablehlo.xor"


StableHLO = Dialect(
    "stablehlo",
    [
        AbsOp,
        AddOp,
        AfterAllOp,
        AndOp,
        Atan2Op,
        BitcastConvertOp,
        CaseOp,
        CbrtOp,
        CeilOp,
        ConstantOp,
        CountLeadingZerosOp,
        MultiplyOp,
        NotOp,
        OrOp,
        PopcntOp,
        ReturnOp,
        ShiftLeftOp,
        ShiftRightArithmeticOp,
        ShiftRightLogicalOp,
        SubtractOp,
        TransposeOp,
        PadOp,
        XorOp,
    ],
    [
        ComparisonDirectionAttr,
        ComparisonTypeAttr,
        DotAttr,
        PrecisionAttr,
        TokenType,
    ],
)
