"""
Unary elementwise operations for the StableHLO dialect.
"""

import abc
from typing import ClassVar, TypeAlias

from xdsl.dialects.builtin import (
    AnyFloat,
    ComplexType,
    IntegerType,
    TensorType,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    result_def,
)

# Type aliases
IntegerTensorType: TypeAlias = TensorType[IntegerType]
FloatOrComplexType: TypeAlias = AnyFloat | ComplexType
FloatOrComplexTensorType: TypeAlias = TensorType[FloatOrComplexType]
FloatTensorType: TypeAlias = TensorType[AnyFloat]


class IntegerTensorLikeElementwiseUnaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", base(IntegerTensorType))

    operand = operand_def(T)
    result = result_def(T)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


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
class PopcntOp(IntegerTensorLikeElementwiseUnaryOperation):
    """
    Performs element-wise count of the number of bits set in the `operand` tensor
    and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt)
    """

    name = "stablehlo.popcnt"
