"""
Type aliases for the StableHLO dialect.
"""

from typing import Literal, TypeAlias

from xdsl.dialects.builtin import (
    I1,
    AnyFloat,
    ComplexType,
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
IntegerTensorType: TypeAlias = TensorType[IntType]
FloatOrComplexType: TypeAlias = AnyFloat | ComplexType
SIntOrFloatOrComplexType: TypeAlias = SIntType | FloatOrComplexType
SIntOrFloatType: TypeAlias = SIntType | AnyFloat
IntOrFloatOrComplexType: TypeAlias = IntType | AnyFloat | ComplexType
FloatOrComplexTensorType: TypeAlias = TensorType[FloatOrComplexType]
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
