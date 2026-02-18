"""
Type aliases for the StableHLO dialect.
"""

from typing import Literal, TypeAlias

from xdsl.dialects.builtin import (
    I1,
    AnyFloat,
    ComplexType,
    Float32Type,
    Float64Type,
    IntegerType,
    Signedness,
    TensorType,
)

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
FloatOrComplexType: TypeAlias = AnyFloat | HLO_ComplexType
SIntOrFloatOrComplexType: TypeAlias = SIntType | FloatOrComplexType
SIntOrFloatType: TypeAlias = SIntType | AnyFloat
IntOrFloatOrComplexType: TypeAlias = IntType | AnyFloat | HLO_ComplexType
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
