"""
Unary elementwise operations for the StableHLO dialect.
"""

import abc
from typing import Generic, TypeAlias, TypeVar

from xdsl.dialects.builtin import (
    AnyFloat,
    AnyTensorType,
    ComplexType,
    IntegerType,
    TensorType,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import NoMemoryEffect

from xdsl_jax.xdsl_extras.traits import (
    Elementwise,
    SameOperandsAndResultShape,
)

from .custom_directives import SameOperandsAndResultType

# Type aliases
IntegerTensorType: TypeAlias = TensorType[IntegerType]
FloatOrComplexType: TypeAlias = AnyFloat | ComplexType
FloatOrComplexTensorType: TypeAlias = TensorType[FloatOrComplexType]
FloatTensorType: TypeAlias = TensorType[AnyFloat]


# Generic type variables for templating
T_IN = TypeVar("T_IN", bound=AnyTensorType)
T_OUT = TypeVar("T_OUT", bound=AnyTensorType)


class ElementwiseUnaryOperation(IRDLOperation, abc.ABC, Generic[T_IN, T_OUT]):
    """
    Templated base class for elementwise unary operations.

    This class provides a flexible template for unary operations that can work
    with different tensor types.

    For more informtation about the semantics, see:
    https://openxla.org/xla/operation_semantics#element-wise_unary_functions
    """

    operand = operand_def(T_IN)
    result = result_def(T_OUT)

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultShape(),
        Elementwise(),
    )

    assembly_format = (
        "$operand attr-dict `:` "
        "custom<SameOperandsAndResultType>(type($operand), type($result))"
    )

    custom_directives = (SameOperandsAndResultType,)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


@irdl_op_definition
class CbrtOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
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
class CeilOp(ElementwiseUnaryOperation[FloatTensorType, FloatTensorType]):
    """
    Performs element-wise ceil of operand tensor and produces a result tensor.
    Implements the roundToIntegralTowardPositive operation from the IEEE-754
    specification.
    For quantized types, performs dequantize_op_quantize(ceil, operand, type(result)).

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#ceil)
    """

    name = "stablehlo.ceil"


@irdl_op_definition
class CountLeadingZerosOp(
    ElementwiseUnaryOperation[IntegerTensorType, IntegerTensorType]
):
    """
    Performs element-wise ceil of `operand` tensor and produces a `result` tensor.
    Implements the `roundToIntegralTowardPositive` operation from the IEEE-754
    specification. For quantized types, performs
    `dequantize_op_quantize(ceil, operand, type(result))`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros)
    """

    name = "stablehlo.count_leading_zeros"


@irdl_op_definition
class FloorOp(ElementwiseUnaryOperation[FloatTensorType, FloatTensorType]):
    """
    Performs element-wise floor of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#floor

    Example:
    ```mlir
    %result = stablehlo.floor %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.floor"


@irdl_op_definition
class NotOp(ElementwiseUnaryOperation[IntegerTensorType, IntegerTensorType]):
    """
    Performs element-wise NOT of tensor `operand` and produces a `result` tensor.
    Depending on the element type, does the following:

    * For booleans: logical NOT.
    * For integers: bitwise NOT.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not)
    """

    name = "stablehlo.not"


@irdl_op_definition
class PopcntOp(ElementwiseUnaryOperation[IntegerTensorType, IntegerTensorType]):
    """
    Performs element-wise count of the number of bits set in the `operand` tensor
    and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt)
    """

    name = "stablehlo.popcnt"


@irdl_op_definition
class RoundNearestAfzOp(ElementwiseUnaryOperation[FloatTensorType, FloatTensorType]):
    """
    Performs element-wise rounding towards the nearest integer, breaking ties
    away from zero, on the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_afz

    Example:
    ```mlir
    %result = stablehlo.round_nearest_afz %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.round_nearest_afz"


@irdl_op_definition
class RoundNearestEvenOp(ElementwiseUnaryOperation[FloatTensorType, FloatTensorType]):
    """
    Performs element-wise rounding towards the nearest integer, breaking ties
    towards the even integer, on the `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_even

    Example:
    ```mlir
    %result = stablehlo.round_nearest_even %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.round_nearest_even"
