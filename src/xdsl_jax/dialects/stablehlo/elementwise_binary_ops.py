"""
Binary elementwise operations for the StableHLO dialect.
"""

import abc
from typing import Generic, TypeVar

from xdsl.dialects.builtin import AnyTensorType
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import Commutative, NoMemoryEffect

from xdsl_jax.xdsl_extras import (
    Elementwise,
    SameOperandsAndResultShape,
    SameOperandsElementType,
)

from .custom_directives import ComplexOpType, SameOperandsAndResultType
from .types import (
    ComplexTensorType,
    Float32Or64TensorType,
    FloatOrComplexTensorType,
    IntegerTensorType,
    IntOrFloatOrComplexTensorType,
    PredOrIntTensorType,
)

# Generic type variables for templating
T = TypeVar("T", bound=AnyTensorType)

# region Abstract Base Classes


class ElementwiseBinaryOperation(IRDLOperation, abc.ABC, Generic[T]):
    """
    Templated base class for elementwise binary operations.

    This class provides a flexible template for binary operations that can work
    with different tensor types.

    For more information about the semantics, see:
    https://openxla.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
    """

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultShape(),
        Elementwise(),
    )

    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` "
        "custom<SameOperandsAndResultType>(type(operands), type(results))"
    )

    custom_directives = (SameOperandsAndResultType,)

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


# endregion


@irdl_op_definition
class AddOp(ElementwiseBinaryOperation[AnyTensorType]):
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

    traits = traits_def(
        Commutative(),
    )


@irdl_op_definition
class AndOp(ElementwiseBinaryOperation[PredOrIntTensorType]):
    """
    Performs element-wise AND of two tensors lhs and rhs and produces a result tensor.
    Depending on the element type, does the following:

    For booleans: logical AND.
    For integers: bitwise AND.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and)
    """

    name = "stablehlo.and"

    traits = traits_def(
        Commutative(),
    )


@irdl_op_definition
class Atan2Op(ElementwiseBinaryOperation[FloatOrComplexTensorType]):
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
class ComplexOp(ElementwiseBinaryOperation[Float32Or64TensorType]):
    """
    Performs element-wise conversion to a complex value from a pair of real and
    imaginary values, `lhs` and `rhs`, and produces a `result` tensor.
    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#complex
    Example:
    ```mlir
    %result = stablehlo.complex %lhs, %rhs : tensor<2xcomplex<f64>>
    ```
    """

    name = "stablehlo.complex"

    result = result_def(ComplexTensorType)

    assembly_format = (
        "$lhs `,` $rhs attr-dict"
        " `:` custom<ComplexOpType>(type(operands), type(results))"
    )

    custom_directives = (ComplexOpType,)

    traits = traits_def(
        SameOperandsElementType(),
    )


@irdl_op_definition
class DivideOp(ElementwiseBinaryOperation[IntOrFloatOrComplexTensorType]):
    """
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

    Example:
    ```mlir
    %result = stablehlo.divide %lhs, %rhs : tensor<4xf32>
    ```
    """

    name = "stablehlo.divide"


@irdl_op_definition
class MaximumOp(ElementwiseBinaryOperation[AnyTensorType]):
    """
    Performs element-wise max operation on tensors `lhs` and `rhs` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum

    Example:
    ```mlir
    %result = stablehlo.maximum %lhs, %rhs : tensor<4xf32>
    ```
    """

    name = "stablehlo.maximum"

    traits = traits_def(
        Commutative(),
    )


@irdl_op_definition
class MinimumOp(ElementwiseBinaryOperation[AnyTensorType]):
    """
    Performs element-wise min operation on tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum

    Example:
    ```mlir
    %result = stablehlo.minimum %lhs, %rhs : tensor<4xf32>
    ```
    """

    name = "stablehlo.minimum"

    traits = traits_def(
        Commutative(),
    )


@irdl_op_definition
class MultiplyOp(ElementwiseBinaryOperation[AnyTensorType]):
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

    traits = traits_def(
        Commutative(),
    )


@irdl_op_definition
class OrOp(ElementwiseBinaryOperation[IntegerTensorType]):
    """
    Performs element-wise OR of two tensors `lhs` and `rhs` and produces a `result`
    tensor. Depending on the element type, does the following:

    * For booleans: logical OR.
    * For integers: bitwise OR.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or)
    """

    name = "stablehlo.or"

    traits = traits_def(
        Commutative(),
    )


@irdl_op_definition
class PowerOp(ElementwiseBinaryOperation[AnyTensorType]):
    """
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

    Example:
    ```mlir
    %result = stablehlo.power %lhs, %rhs : tensor<6xf64>
    ```
    """

    name = "stablehlo.power"


@irdl_op_definition
class RemainderOp(ElementwiseBinaryOperation[AnyTensorType]):
    """
    Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder

    Example:
    ```mlir
    %result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
    ```
    """

    name = "stablehlo.remainder"


@irdl_op_definition
class ShiftLeftOp(ElementwiseBinaryOperation[IntegerTensorType]):
    """
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs` number
    of bits and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left)
    """

    name = "stablehlo.shift_left"


@irdl_op_definition
class ShiftRightArithmeticOp(ElementwiseBinaryOperation[IntegerTensorType]):
    """
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor by
    `rhs` number of bits and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic)
    """

    name = "stablehlo.shift_right_arithmetic"


@irdl_op_definition
class ShiftRightLogicalOp(ElementwiseBinaryOperation[IntegerTensorType]):
    """
    Performs element-wise logical right-shift operation on the `lhs` tensor by `rhs`
    number of bits and produces a `result` tensor.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical)
    """

    name = "stablehlo.shift_right_logical"


@irdl_op_definition
class SubtractOp(ElementwiseBinaryOperation[IntOrFloatOrComplexTensorType]):
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
class XorOp(ElementwiseBinaryOperation[IntegerTensorType]):
    """
    Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a `result`
    tensor. Depending on the element type, does the following:

    * For booleans: logical XOR.
    * For integers: bitwise XOR.

    [See StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor)
    """

    name = "stablehlo.xor"

    traits = traits_def(
        Commutative(),
    )
