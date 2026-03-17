"""
[StableHLO](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
is an operation set for high-level operations (HLO) in machine learning (ML) models.
StableHLO works as a portability layer between different ML frameworks and ML compilers:
ML frameworks that produce StableHLO programs are compatible with ML compilers that
consume StableHLO programs.
"""

from xdsl.ir import Dialect

from .attributes import (
    ComparisonDirectionAttr,
    ComparisonTypeAttr,
    DotAlgorithmAttr,
    DotAttr,
    GatherDimensionNumbers,
    OutputOperandAlias,
    PrecisionAttr,
    ResultAccuracyModeAttr,
    ScatterDimensionNumbers,
    TokenType,
)
from .control_flow_ops import AfterAllOp, CaseOp, IfOp, OptimizationBarrierOp, WhileOp
from .data_movement_ops import (
    BroadcastInDimOp,
    ConcatenateOp,
    DynamicSliceOp,
    GatherOp,
    PadOp,
    ReshapeOp,
    ScatterOp,
    SliceOp,
    TransposeOp,
)
from .dynamism_ops import DynamicBroadcastInDimOp
from .elementwise_binary_ops import (
    AddOp,
    AndOp,
    Atan2Op,
    ComplexOp,
    DivideOp,
    MaximumOp,
    MinimumOp,
    MultiplyOp,
    OrOp,
    PowerOp,
    RemainderOp,
    ShiftLeftOp,
    ShiftRightArithmeticOp,
    ShiftRightLogicalOp,
    SubtractOp,
    XorOp,
)
from .elementwise_other_ops import (
    BitcastConvertOp,
    ClampOp,
    CompareOp,
    MapOp,
    ReducePrecisionOp,
    SelectOp,
)
from .elementwise_unary_ops import (
    AbsOp,
    CbrtOp,
    CeilOp,
    ConvertOp,
    CosineOp,
    CountLeadingZerosOp,
    ExponentialMinusOneOp,
    ExponentialOp,
    FloorOp,
    ImagOp,
    IsFiniteOp,
    LogisticOp,
    LogOp,
    LogPlusOneOp,
    NegateOp,
    NotOp,
    PopcntOp,
    RealOp,
    RoundNearestAfzOp,
    RoundNearestEvenOp,
    RsqrtOp,
    SignOp,
    SineOp,
    SqrtOp,
    TanhOp,
    TanOp,
)
from .extensibility_ops import CustomCallOp
from .miscellaneous_ops import ConstantOp, IotaOp
from .modularity_ops import (
    ReturnOp,
)
from .reduction_ops import DotGeneralOp, ReduceOp

StableHLO = Dialect(
    "stablehlo",
    [
        # Elementwise unary operations
        AbsOp,
        BitcastConvertOp,
        CbrtOp,
        CeilOp,
        ConvertOp,
        CosineOp,
        CountLeadingZerosOp,
        ExponentialMinusOneOp,
        ExponentialOp,
        FloorOp,
        ImagOp,
        IsFiniteOp,
        LogisticOp,
        LogOp,
        LogPlusOneOp,
        NegateOp,
        NotOp,
        PopcntOp,
        RealOp,
        RoundNearestAfzOp,
        RoundNearestEvenOp,
        RsqrtOp,
        SineOp,
        SignOp,
        SqrtOp,
        TanOp,
        TanhOp,
        # Elementwise binary operations
        AddOp,
        AndOp,
        Atan2Op,
        ComplexOp,
        DivideOp,
        MaximumOp,
        MinimumOp,
        MultiplyOp,
        OrOp,
        PowerOp,
        RemainderOp,
        ShiftLeftOp,
        ShiftRightArithmeticOp,
        ShiftRightLogicalOp,
        SubtractOp,
        XorOp,
        # Elementwise other operations
        ClampOp,
        CompareOp,
        MapOp,
        ReducePrecisionOp,
        SelectOp,
        # Control flow operations
        AfterAllOp,
        CaseOp,
        IfOp,
        OptimizationBarrierOp,
        WhileOp,
        # Data movement operations
        BroadcastInDimOp,
        ConcatenateOp,
        DynamicSliceOp,
        GatherOp,
        PadOp,
        ReshapeOp,
        ScatterOp,
        SliceOp,
        TransposeOp,
        # Dynamism operations
        DynamicBroadcastInDimOp,
        # Reduction operations
        DotGeneralOp,
        ReduceOp,
        # Extensibility operations
        CustomCallOp,
        # Miscellaneous operations
        ConstantOp,
        IotaOp,
        ReturnOp,
    ],
    [
        ComparisonDirectionAttr,
        ComparisonTypeAttr,
        DotAlgorithmAttr,
        DotAttr,
        GatherDimensionNumbers,
        OutputOperandAlias,
        PrecisionAttr,
        ResultAccuracyModeAttr,
        ScatterDimensionNumbers,
        TokenType,
    ],
)

__all__ = ["StableHLO"]
