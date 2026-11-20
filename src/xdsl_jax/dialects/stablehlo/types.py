"""
Type aliases for the StableHLO dialect.
"""

from typing import Literal

from xdsl.dialects.builtin import (
    I1,
    I32,
    AnyFloat,
    AnyTensorType,
    ArrayAttr,
    ComplexType,
    Float32Type,
    Float64Type,
    IndexType,
    IntAttrConstraint,
    IntegerType,
    MemRefType,
    Signedness,
    TensorType,
)
from xdsl.ir import Attribute
from xdsl.irdl import (
    AnyInt,
    EqIntConstraint,
    IRDLAttrConstraint,
    RangeLengthConstraint,
    RangeOf,
    eq,
)

from .attributes import TokenType

type SIntType = IntegerType[
    Literal[1, 2, 4, 8, 16, 32, 64],
    Literal[Signedness.SIGNLESS],
]
# NOTE: IntegerType is defined in the StableHLO spec as:
# IntegerType ::= SignedIntegerType | UnsignedIntegerType,
# but the MLIR implementation is using signless integers instead of signed,
# and there is a TODO to fix it.
type IntType = IntegerType[
    Literal[1, 2, 4, 8, 16, 32, 64],
    Literal[Signedness.UNSIGNED, Signedness.SIGNLESS],
]
type Float32Or64Type = Float32Type | Float64Type
type HLO_ComplexType = ComplexType[Float32Or64Type]
type ComplexTensorType = TensorType[HLO_ComplexType]
type IntegerTensorType = TensorType[IntType]
# TODO: Change to SI32 once StableHLO adopts signful integer semantics
# See: https://github.com/openxla/stablehlo/issues/22
# https://github.com/openxla/stablehlo/issues/2489
type SI32TensorType = TensorType[I32]
type FloatOrComplexType = AnyFloat | HLO_ComplexType
type SIntOrFloatOrComplexType = SIntType | FloatOrComplexType
type SIntOrFloatType = SIntType | AnyFloat
type IntOrFloatOrComplexType = IntType | AnyFloat | HLO_ComplexType
type BufferType = MemRefType[Attribute]
type FloatOrComplexTensorType = TensorType[FloatOrComplexType]
type Float32Or64TensorType = TensorType[Float32Or64Type]
type FloatTensorType = TensorType[AnyFloat]
type PredTensorType = TensorType[I1]
type SIntOrFloatOrComplexTensorType = TensorType[SIntOrFloatOrComplexType]
type SIntOrFloatTensorType = TensorType[SIntOrFloatType]
type PredOrIntType = IntegerType[
    Literal[1, 2, 4, 8, 16, 32, 64],
    Literal[Signedness.UNSIGNED, Signedness.SIGNLESS],
]
type PredOrIntTensorType = TensorType[PredOrIntType]
type IntOrFloatOrComplexTensorType = TensorType[IntOrFloatOrComplexType]
type TensorOrTokenType = AnyTensorType | TokenType
type TensorOrTokenOrBufferType = AnyTensorType | TokenType | BufferType
type DimensionValueType = IntType | IndexType
# DimensionTensorType is a 1D tensor of dimension values
DimensionTensorType: IRDLAttrConstraint = TensorType.constr(
    element_type=DimensionValueType,
    shape=ArrayAttr.constr(
        RangeLengthConstraint(
            constraint=RangeOf(IntAttrConstraint(AnyInt())),
            length=EqIntConstraint(1),
        )
    ),
)
type IntegerOrIndexTensorType = TensorType[IntegerType | IndexType]

# ScalarIntTensorType is a 0D tensor of integer values
ScalarIntTensorType: IRDLAttrConstraint = TensorType.constr(
    element_type=IntType,
    shape=eq(ArrayAttr(())),
)
