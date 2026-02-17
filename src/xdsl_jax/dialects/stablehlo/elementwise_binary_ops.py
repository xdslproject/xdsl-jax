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
from xdsl.traits import NoMemoryEffect

from xdsl_jax.xdsl_extras import (
    Elementwise,
    SameOperandsAndResultShape,
)

from .custom_directives import SameOperandsAndResultType
from .types import (
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
