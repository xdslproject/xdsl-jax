"""
Type aliases for the StableHLO dialect.
"""

from typing import Literal, TypeAlias

from xdsl.dialects.builtin import (
    I1,
    I32,
    AnyFloat,
    AnyTensorType,
    ComplexType,
    Float32Type,
    Float64Type,
    IntegerType,
    MemRefType,
    Signedness,
    TensorType,
)
from xdsl.ir import Attribute

from .attributes import TokenType

SIntType: TypeAlias = IntegerType[
    Literal[2, 4, 8, 16, 32, 64],
    Literal[Signedness.SIGNLESS],
]
# NOTE: IntegerType is defined in the StableHLO spec as:
# IntegerType ::= SignedIntegerType | UnsignedIntegerType,
# but the MLIR implementation is using signless integers instead of signed,
# and there is a TODO to fix it.
IntType: TypeAlias = IntegerType[
    Literal[2, 4, 8, 16, 32, 64],
    Literal[Signedness.UNSIGNED, Signedness.SIGNLESS],
]
Float32Or64Type: TypeAlias = Float32Type | Float64Type
HLO_ComplexType: TypeAlias = ComplexType[Float32Or64Type]
ComplexTensorType: TypeAlias = TensorType[HLO_ComplexType]
IntegerTensorType: TypeAlias = TensorType[IntType]
# TODO: Change to SI32 once StableHLO adopts signful integer semantics
# See: https://github.com/openxla/stablehlo/issues/22
# https://github.com/openxla/stablehlo/issues/2489
SI32TensorType: TypeAlias = TensorType[I32]
FloatOrComplexType: TypeAlias = AnyFloat | HLO_ComplexType
SIntOrFloatOrComplexType: TypeAlias = SIntType | FloatOrComplexType
SIntOrFloatType: TypeAlias = SIntType | AnyFloat
IntOrFloatOrComplexType: TypeAlias = IntType | AnyFloat | HLO_ComplexType
BufferType: TypeAlias = MemRefType[Attribute]
FloatOrComplexTensorType: TypeAlias = TensorType[FloatOrComplexType]
Float32Or64TensorType: TypeAlias = TensorType[Float32Or64Type]
FloatTensorType: TypeAlias = TensorType[AnyFloat]
PredTensorType: TypeAlias = TensorType[I1]
SIntOrFloatOrComplexTensorType: TypeAlias = TensorType[SIntOrFloatOrComplexType]
SIntOrFloatTensorType: TypeAlias = TensorType[SIntOrFloatType]
PredOrIntType: TypeAlias = IntegerType[
    Literal[1, 2, 4, 8, 16, 32, 64],
    Literal[Signedness.UNSIGNED, Signedness.SIGNLESS],
]
PredOrIntTensorType: TypeAlias = TensorType[PredOrIntType]
IntOrFloatOrComplexTensorType: TypeAlias = TensorType[IntOrFloatOrComplexType]
TensorOrTokenType: TypeAlias = AnyTensorType | TokenType
TensorOrTokenOrBufferType: TypeAlias = AnyTensorType | TokenType | BufferType
