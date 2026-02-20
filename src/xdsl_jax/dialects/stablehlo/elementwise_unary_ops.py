"""
Unary elementwise operations for the StableHLO dialect.
"""

import abc
from typing import Generic, TypeVar

from xdsl.dialects.builtin import AnyTensorType
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import NoMemoryEffect

from xdsl_jax.xdsl_extras.traits import (
    Elementwise,
    SameOperandsAndResultShape,
)

from .attributes import ResultAccuracyMode, ResultAccuracyModeAttr
from .custom_directives import SameOperandsAndResultType
from .types import (
    FloatOrComplexTensorType,
    FloatTensorType,
    IntegerTensorType,
    IntOrFloatOrComplexTensorType,
    PredTensorType,
    SIntOrFloatOrComplexTensorType,
    SIntOrFloatTensorType,
)

# Generic type variables for templating
T_OPERAND = TypeVar("T_OPERAND", bound=AnyTensorType)
T_RESULT = TypeVar("T_RESULT", bound=AnyTensorType)


class ElementwiseUnaryOperation(IRDLOperation, abc.ABC, Generic[T_OPERAND, T_RESULT]):
    """
    Templated base class for elementwise unary operations.

    This class provides a flexible template for unary operations that can work
    with different tensor types.

    For more informtation about the semantics, see:
    https://openxla.org/xla/operation_semantics#element-wise_unary_functions
    """

    operand = operand_def(T_OPERAND)
    result = result_def(T_RESULT)

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultShape(),
        Elementwise(),
    )

    assembly_format = (
        "$operand attr-dict `:` "
        "custom<SameOperandsAndResultType>(type(operands), type(results))"
    )

    custom_directives = (SameOperandsAndResultType,)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


@irdl_op_definition
class AbsOp(
    ElementwiseUnaryOperation[SIntOrFloatOrComplexTensorType, SIntOrFloatTensorType]
):
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

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


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
class ConvertOp(
    ElementwiseUnaryOperation[
        IntOrFloatOrComplexTensorType, IntOrFloatOrComplexTensorType
    ]
):
    """
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xcomplex<f64>>
    ```
    """

    name = "stablehlo.convert"

    traits = traits_def(SameOperandsAndResultShape())


@irdl_op_definition
class CosineOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise cosine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cosine

    Example:
    ```mlir
    %result = stablehlo.cosine %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.cosine"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


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
class ExponentialMinusOneOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise exponential minus one operation on `operand` tensor
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one

    Example:
    ```mlir
    %result = stablehlo.exponential_minus_one %operand : tensor<2xf64>
    ```
    """

    name = "stablehlo.exponential_minus_one"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class ExponentialOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise exponential operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential

    Example:
    ```mlir
    %result = stablehlo.exponential %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.exponential"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


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
class ImagOp(ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatTensorType]):
    """
    Extracts the imaginary part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#imag

    Example:
    ```mlir
    %result = stablehlo.imag %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    ```
    """

    name = "stablehlo.imag"


@irdl_op_definition
class IsFiniteOp(ElementwiseUnaryOperation[FloatTensorType, PredTensorType]):
    """
    Performs element-wise check whether the value in `x` is finite (i.e. is
    neither +Inf, -Inf, nor NaN) and produces a `y` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#is_finite

    Example:
    ```mlir
    %y = stablehlo.is_finite %x : (tensor<7xf64>) -> tensor<7xi1>
    ```
    """

    name = "stablehlo.is_finite"


@irdl_op_definition
class LogisticOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise logistic operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#logistic

    Example:
    ```mlir
    %result = stablehlo.logistic %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.logistic"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class LogOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise logarithm operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log

    Example:
    ```mlir
    %result = stablehlo.log %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.log"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class LogPlusOneOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise logarithm plus one operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one

    Example:
    ```mlir
    %result = stablehlo.log_plus_one %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.log_plus_one"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class NegateOp(
    ElementwiseUnaryOperation[
        IntOrFloatOrComplexTensorType, IntOrFloatOrComplexTensorType
    ]
):
    """
    Performs element-wise negation of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

    Example:
    ```mlir
    %result = stablehlo.negate %operand : tensor<2x3xi32>
    ```
    """

    name = "stablehlo.negate"


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
class RealOp(ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatTensorType]):
    """
    Extracts the real part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#real

    Example:
    ```mlir
    %result = stablehlo.real %operand : tensor<2xcomplex<f32>> : tensor<2xf32>
    ```
    """

    name = "stablehlo.real"


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


@irdl_op_definition
class RsqrtOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise reciprocal square root operation on `operand` tensor
    and produces a `result` tensor, implementing the `rSqrt` operation from the
    IEEE-754 specification.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rsqrt

    Example:
    ```mlir
    %result = stablehlo.rsqrt %operand : tensor<2x2xf32>
    ```
    """

    name = "stablehlo.rsqrt"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class SignOp(
    ElementwiseUnaryOperation[
        SIntOrFloatOrComplexTensorType, SIntOrFloatOrComplexTensorType
    ]
):
    """
    Returns the sign of the `operand` element-wise and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    Example:
    ```mlir
    %result = stablehlo.sign %operand : tensor<5xf64>
    ```
    """

    name = "stablehlo.sign"


@irdl_op_definition
class SineOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise sine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sine

    Example:
    ```mlir
    %result = stablehlo.sine %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.sine"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class SqrtOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise square root operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sqrt

    Example:
    ```mlir
    %result = stablehlo.sqrt %operand : tensor<2x2xf32>
    ```
    """

    name = "stablehlo.sqrt"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class TanOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise tangent operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tan

    Example:
    ```mlir
    %result = stablehlo.tan %operand : tensor<2x2xf64>
    ```
    """

    name = "stablehlo.tan"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)


@irdl_op_definition
class TanhOp(
    ElementwiseUnaryOperation[FloatOrComplexTensorType, FloatOrComplexTensorType]
):
    """
    Performs element-wise hyperbolic tangent operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tanh

    Example:
    ```mlir
    %result = stablehlo.tanh %operand : tensor<2xf32>
    ```
    """

    name = "stablehlo.tanh"

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)
