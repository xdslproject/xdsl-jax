"""Modularity operations for the StableHLO dialect."""

from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, traits_def, var_operand_def
from xdsl.traits import IsTerminator, Pure

from .types import TensorOrTokenOrBufferType


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
