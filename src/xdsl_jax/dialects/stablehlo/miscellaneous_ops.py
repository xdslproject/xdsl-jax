"""
Miscellaneous operations for the StableHLO dialect.
"""

from typing import cast

from xdsl.dialects.builtin import (
    AnyTensorType,
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    TensorType,
    i64,
)
from xdsl.ir import Attribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.irdl.attributes import eq
from xdsl.irdl.constraints import AtLeast
from xdsl.traits import ConstantLike, Pure
from xdsl.utils.exceptions import VerifyException

from .custom_directives import ConstantOpValue
from .types import IntOrFloatOrComplexTensorType


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """
    Produces an `output` tensor from a constant `value`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant)
    """

    name = "stablehlo.constant"

    value = prop_def(DenseIntOrFPElementsAttr)
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
