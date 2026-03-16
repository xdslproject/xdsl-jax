"""
Other elementwise operations for the StableHLO dialect.
"""

from typing import cast

from xdsl.dialects.builtin import (
    AnyTensorType,
    DenseArrayBase,
    IntegerAttr,
    TensorType,
    i32,
    i64,
)
from xdsl.ir import Attribute, Block
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
)
from xdsl.irdl.attributes import eq
from xdsl.irdl.constraints import AtLeast
from xdsl.traits import (
    ConditionallySpeculatable,
    NoMemoryEffect,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException

from xdsl_jax.xdsl_extras.traits import Elementwise, SameOperandsAndResultShape

from .attributes import (
    ComparisonDirectionAttr,
    ComparisonTypeAttr,
    ResultAccuracyMode,
    ResultAccuracyModeAttr,
)
from .custom_directives import ExponentMantissa, SameOperandsAndResultType, SelectOpType
from .modularity_ops import ReturnOp
from .traits import (
    CompatibleOperandsAndResultType,
    SpeculatableIfAllInputsStatic,
    SpeculatableIfStaticDimInOutputIsStaticInInput,
)
from .types import FloatTensorType, PredTensorType


@irdl_op_definition
class ClampOp(IRDLOperation):
    """Element-wise clamp with min and max bounds."""

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
class BitcastConvertOp(IRDLOperation):
    """
    Performs a bitcast operation on operand tensor and produces a result tensor
    where the bits of the entire operand tensor are reinterpreted using the type of the
    result tensor.
    """

    name = "stablehlo.bitcast_convert"
    input = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    traits = traits_def(NoMemoryEffect(), ConditionallySpeculatable())


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
    comparison_direction = prop_def(ComparisonDirectionAttr)
    comparison_type = opt_prop_def(ComparisonTypeAttr)

    traits = traits_def(
        NoMemoryEffect(),
        Elementwise(),
        SameOperandsAndResultShape(),
    )


@irdl_op_definition
class MapOp(IRDLOperation):
    """
    Applies a map function `computation` to `inputs` along the `dimensions` and
    produces a `result` tensor.
    """

    name = "stablehlo.map"

    inputs = var_operand_def(AnyTensorType)
    dimensions = prop_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)
    computation = var_region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        SameOperandsAndResultShape(),
        SingleBlockImplicitTerminator(ReturnOp),
    )

    def _verify_computation_block(self, block: Block) -> None:
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
class SelectOp(IRDLOperation):
    """
    Produces a `result` tensor where each element is selected from `on_true` or
    `on_false` tensor based on the value of the corresponding element of `pred`.
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


@irdl_op_definition
class ReducePrecisionOp(IRDLOperation):
    """
    Performs element-wise conversion of `operand` to another floating-point type
    that uses `exponent_bits` and `mantissa_bits` and back to the original
    floating-point type and produces an `output` tensor.
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

    exponent_bits = prop_def(IntegerAttr.constr(type=eq(i32), value=AtLeast(1)))
    mantissa_bits = prop_def(IntegerAttr.constr(type=eq(i32), value=AtLeast(0)))

    traits = traits_def(
        NoMemoryEffect(),
        Elementwise(),
        CompatibleOperandsAndResultType(),
        SpeculatableIfStaticDimInOutputIsStaticInInput(),
    )

    result_accuracy = opt_prop_def(
        ResultAccuracyModeAttr, ResultAccuracyModeAttr(ResultAccuracyMode.DEFAULT)
    )

    irdl_options = (ParsePropInAttrDict(),)
