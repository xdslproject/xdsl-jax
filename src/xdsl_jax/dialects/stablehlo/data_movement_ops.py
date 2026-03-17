"""
Data movement operations for the StableHLO dialect.
"""

from typing import cast

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnyTensorType,
    BoolAttr,
    DenseArrayBase,
    IntegerAttr,
    TensorType,
    i64,
)
from xdsl.interfaces import ConditionallySpeculatableInterface
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    AtLeast,
    IRDLOperation,
    SameVariadicOperandSize,
    eq,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    ConditionallySpeculatable,
    NoMemoryEffect,
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.type import get_element_type_or_self

from xdsl_jax.xdsl_extras import (
    AllMatchSameOperatorTrait,
    SameOperandsAndResultElementType,
)

from .attributes import GatherDimensionNumbers, ScatterDimensionNumbers
from .custom_directives import SliceRanges, VariadicOperandWithAttribute
from .traits import (
    SpeculatableIfAllInputsStatic,
    SpeculatableIfStaticDimInOutputIsStaticInInput,
)
from .types import (
    IntegerOrIndexTensorType,
    IntegerTensorType,
    ScalarIntTensorType,
    SI32TensorType,
)


@irdl_op_definition
class GatherOp(IRDLOperation, ConditionallySpeculatableInterface):
    """
    Gathers slices from ``operand`` tensor from offsets specified in
    ``start_indices`` and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

    Example:
    ```mlir
    %result = "stablehlo.gather"(%operand, %start_indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [3, 4],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [1],
        start_index_map = [2, 1],
        index_vector_dim = 3>,
      slice_sizes = array<i64: 1, 1, 2, 2>,
      indices_are_sorted = false
    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi64>
    ```
    """

    name = "stablehlo.gather"
    operand = operand_def(AnyTensorType)
    start_indices = operand_def(IntegerTensorType)
    dimension_numbers = prop_def(GatherDimensionNumbers)
    slice_sizes = prop_def(DenseArrayBase.constr(i64))
    indices_are_sorted = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    result = result_def(AnyTensorType)

    traits = traits_def(
        NoMemoryEffect(),
        AllMatchSameOperatorTrait(
            ("operand", "result"),
            lambda x: get_element_type_or_self(x.type),
            "element type",
        ),
    )

    def is_speculatable(self) -> bool:
        if self.indices_are_sorted:
            return False
        return all(
            isinstance(t, TensorType) and t.has_static_shape()
            for t in self.operand_types
        )


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Permutes the dimensions of `operand` tensor using `permutation` and produces a
    `result` tensor. More formally, `result[result_index] = operand[operand_index]`
    where `result_index[d] = operand_index[permutation[d]]`.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose)
    """

    name = "stablehlo.transpose"

    operand = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)
    permutation = prop_def(DenseArrayBase.constr(i64))

    traits = traits_def(NoMemoryEffect(), ConditionallySpeculatable())

    assembly_format = (
        "$operand `,` `dims` `=` $permutation "
        "attr-dict `:` functional-type(operands, results)"
    )

    def __init__(
        self, operand: SSAValue, permutation: DenseArrayBase, result_type: Attribute
    ):
        super().__init__(
            operands=(operand,),
            result_types=(result_type,),
            attributes={"permutation": permutation},
        )

    def get_permutation(self) -> tuple[int, ...]:
        return self.permutation.get_values()

    def verify_(self) -> None:
        # Operand and result types are checked before the custom `verify_`
        o_type = cast(TensorType[Attribute], self.operand.type)
        r_type = self.result.type

        o_shape = o_type.get_shape()
        r_shape = r_type.get_shape()

        # `permutation` is a permutation of `range(rank(operand))`
        permutation = self.get_permutation()
        if sorted(permutation) != list(range(len(o_shape))):
            raise VerifyException(
                f"Permutation {permutation} of transpose must be a permutation of "
                f"range({len(o_shape)})"
            )

        # `shape(result) = dim(operand, permutation...)`
        for i, dim in enumerate(permutation):
            if r_shape[i] != o_shape[dim]:
                raise VerifyException(
                    f"Permutation mismatch at dimension {i}, expected {o_shape[dim]}"
                )


@irdl_op_definition
class PadOp(IRDLOperation):
    """
    Expands operand by padding around the tensor as well as between the
    elements of the tensor with the given padding_value.

    edge_padding_low and edge_padding_high specify the amount of padding
    added at the low-end (next to index 0) and the high-end
    (next to the highest index) of each dimension respectively.
    The amount of padding can be negative, where the absolute value of negative
    padding indicates the number of elements to remove from the specified dimension.

    interior_padding specifies the amount of padding added between any
    two elements in each dimension which may not be negative. Interior padding occurs
    before edge padding such that negative edge padding will remove elements from
    the interior-padded operand.

    More formally, result[result_index] is defined as:

    operand[operand_index] if
    result_index = edge_padding_low + operand_index * (interior_padding + 1).
    padding_value otherwise.

    See [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad)
    """

    name = "stablehlo.pad"

    operand = operand_def(AnyTensorType)
    padding_value = operand_def(SI32TensorType)
    result = result_def(AnyTensorType)
    edge_padding_low = prop_def(DenseArrayBase.constr(i64))
    edge_padding_high = prop_def(DenseArrayBase.constr(i64))
    interior_padding = prop_def(DenseArrayBase.constr(i64))

    traits = traits_def(NoMemoryEffect(), SameOperandsAndResultElementType())

    assembly_format = (
        "$operand `,` $padding_value `,` "
        "`low` `=` $edge_padding_low `,` "
        "`high` `=` $edge_padding_high `,` "
        "`interior` `=` $interior_padding "
        "attr-dict `:` functional-type(operands, results)"
    )

    def __init__(
        self,
        operand: SSAValue,
        padding_value: SSAValue,
        edge_padding_low: DenseArrayBase,
        edge_padding_high: DenseArrayBase,
        interior_padding: DenseArrayBase,
        result_type: Attribute,
    ):
        super().__init__(
            operands=(operand, padding_value),
            result_types=(result_type,),
            attributes={
                "edge_padding_low": edge_padding_low,
                "edge_padding_high": edge_padding_high,
                "interior_padding": interior_padding,
            },
        )

    def get_edge_padding_low(self) -> tuple[int, ...]:
        return self.edge_padding_low.get_values()

    def get_edge_padding_high(self) -> tuple[int, ...]:
        return self.edge_padding_high.get_values()

    def get_interior_padding(self) -> tuple[int, ...]:
        return self.interior_padding.get_values()

    def verify_(self) -> None:
        # Operand and result types are checked before the custom `verify_`
        o_type = cast(TensorType[Attribute], self.operand.type)
        pad_val_type = cast(TensorType[Attribute], self.padding_value.type)
        r_type = self.result.type

        o_shape = o_type.get_shape()
        pad_val_shape = pad_val_type.get_shape()
        r_shape = r_type.get_shape()

        if pad_val_shape:
            raise VerifyException(
                f"Expect padding_value is an 0-dimensional tensor,"
                f" found {pad_val_shape}"
            )

        o_rank = len(o_shape)
        edge_padding_low = self.get_edge_padding_low()
        edge_padding_high = self.get_edge_padding_high()
        interior_padding = self.get_interior_padding()

        hints = [
            "result shape",
            "edge_padding_low",
            "edge_padding_high",
            "interior_padding",
        ]

        # size(edge_padding_low) = size(edge_padding_high)
        # = size(interior_padding) = rank(operand)
        for shape, hint in zip(
            [r_shape, edge_padding_low, edge_padding_high, interior_padding], hints
        ):
            if o_rank != len(shape):
                raise VerifyException(
                    "Pad operation rank mismatch while "
                    f"the operand has {o_rank} dimension(s) and "
                    f"{hint} has {len(shape)} dimension(s)"
                )

        # 0 <= interior_padding
        for inner_padding in interior_padding:
            if inner_padding < 0:
                raise VerifyException(
                    f"The interior_padding value must be equal or larger than 0,"
                    f" found {inner_padding} "
                )

        # shape(result) = shape(operand) + edge_padding_low +
        # max(shape(operand) - 1, 0) * interior_padding + edge_padding_high
        for ith_dim, (r_dim, o_dim, pad_low, pad_high, inner_pad) in enumerate(
            zip(r_shape, o_shape, edge_padding_low, edge_padding_high, interior_padding)
        ):
            if r_dim != o_dim + pad_low + max(o_dim - 1, 0) * inner_pad + pad_high:
                raise VerifyException(
                    f"Pad operation at {ith_dim} dimension  mismatch, while "
                    f"the dimension before {o_dim} and the after is {r_dim} "
                    f"with {pad_low}, {pad_high}, {inner_pad} "
                    f"as low, high and inner padding values"
                )


@irdl_op_definition
class ScatterOp(IRDLOperation, ConditionallySpeculatableInterface):
    """
     Produces ``results`` tensors which are equal to ``inputs`` tensors except that
     several slices specified by ``scatter_indices`` are updated with the values
     ``updates`` using ``update_computation``.

     See:
     https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

    Example:
    ```mlir
    %result = "stablehlo.scatter"(%input, %scatter_indices, %update) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [3, 4],
        inserted_window_dims = [1],
        input_batching_dims = [0],
        scatter_indices_batching_dims = [1],
        scatter_dims_to_operand_dims = [2, 1],
        index_vector_dim = 3>,
      indices_are_sorted = false,
      unique_indices = false
    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>)
        -> tensor<2x3x4x2xi64>
    ```
    """

    name = "stablehlo.scatter"
    inputs = var_operand_def(AnyTensorType)
    scatter_indices = operand_def(IntegerOrIndexTensorType)
    updates = var_operand_def(AnyTensorType)
    scatter_dimension_numbers = prop_def(ScatterDimensionNumbers)
    indices_are_sorted = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    unique_indices = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    result = var_result_def(AnyTensorType)
    update_computation = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
    )

    irdl_options = (SameVariadicOperandSize(),)

    def is_speculatable(self) -> bool:
        if self.unique_indices.value.data or self.indices_are_sorted.value.data:
            return False
        if not all(
            isinstance(t, TensorType) and t.has_static_shape()
            for t in self.operand_types
        ):
            return False
        return RecursivelySpeculatable.is_speculatable(self)

    # TODO: Implement custom verifier for the scatter operation.


@irdl_op_definition
class ConcatenateOp(IRDLOperation, ConditionallySpeculatableInterface):
    """
    Concatenates a variadic number of tensors in ``inputs`` along ``dimension``
    dimension in the same order as the given arguments and produces a ``result``
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = stablehlo.concatenate %input0, %input1, dim = 0
    : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
    """

    name = "stablehlo.concatenate"

    inputs = var_operand_def(AnyTensorType)
    result = result_def(AnyTensorType)
    dimension = prop_def(IntegerAttr.constr(type=eq(i64), value=AtLeast(0)))

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultElementType(),
    )

    assembly_format = (
        "custom<VariadicOperandWithAttribute>($inputs) "
        "`dim` `=` $dimension attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (VariadicOperandWithAttribute,)

    def is_speculatable(self) -> bool:
        if not self.operands or not self.results:
            return False

        concat_dim = self.dimension.value.data
        result_shape = cast(TensorType, self.result_types[0]).get_shape()
        concat_dim_dynamic = result_shape[concat_dim] == DYNAMIC_INDEX
        for operand_type in self.operand_types:
            operand_shape = cast(TensorType, operand_type).get_shape()
            for idx, dim in enumerate(operand_shape):
                if idx == concat_dim and concat_dim_dynamic:
                    continue
                if dim == DYNAMIC_INDEX:
                    return False

        return True


@irdl_op_definition
class DynamicSliceOp(IRDLOperation):
    """
    Extracts a slice from the ``operand`` using dynamically-computed starting
    indices and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

    Example:
    ```mlir
    %result = stablehlo.dynamic_slice %operand, %start_indices0, %start_indices1,
      sizes = [2, 2] : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
    ```
    """

    name = "stablehlo.dynamic_slice"
    operand = operand_def(AnyTensorType)
    start_indices = var_operand_def(ScalarIntTensorType)
    slice_sizes = prop_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)

    traits = traits_def(
        Pure(),
        AllMatchSameOperatorTrait(
            ("operand", "result"),
            lambda x: get_element_type_or_self(x.type),
            "element type",
        ),
    )

    assembly_format = (
        "$operand `,` custom<VariadicOperandWithAttribute>($start_indices) "
        "`sizes` `=` $slice_sizes attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (VariadicOperandWithAttribute,)


@irdl_op_definition
class BroadcastInDimOp(IRDLOperation):
    """
    Expands the dimensions and/or rank of an input tensor by duplicating the
    data in the ``operand`` tensor and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    Example:
    ```mlir
    %result = stablehlo.broadcast_in_dim %operand, dims = [2, 1]
      : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    ```
    """

    name = "stablehlo.broadcast_in_dim"
    operand = operand_def(AnyTensorType)
    broadcast_dimensions = prop_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)

    assembly_format = """
        $operand `,` `dims` `=` $broadcast_dimensions
          attr-dict `:` functional-type(operands, results)
    """

    traits = traits_def(
        NoMemoryEffect(),
        SpeculatableIfAllInputsStatic(),
        SameOperandsAndResultElementType(),
    )

    def verify_(self) -> None:
        """Verify non-quantized broadcast_in_dim constraints."""
        o_type = self.operands[0].type
        r_type = self.result.type

        assert isinstance(o_type, TensorType)
        assert isinstance(r_type, TensorType)

        if not r_type.has_static_shape():
            raise VerifyException("broadcast_in_dim output must have a static shape.")

        # (C2) broadcast_dimensions size == operand rank
        dims = tuple(self.broadcast_dimensions.get_values())
        operand_rank = o_type.get_num_dims()
        if len(dims) != operand_rank:
            raise VerifyException(
                "broadcast_dimensions size ("
                f"{len(dims)}"
                ") does not match operand rank ("
                f"{operand_rank}"
                ")"
            )

        # (C4) broadcast_dimensions should not have duplicates
        if len(set(dims)) != len(dims):
            raise VerifyException("broadcast_dimensions should not have duplicates")

        result_rank = r_type.get_num_dims()
        o_shape = o_type.get_shape()
        r_shape = r_type.get_shape()

        for i, dim_index in enumerate(dims):
            # (C3) each dim index in bounds of result rank
            if dim_index < 0 or dim_index >= result_rank:
                raise VerifyException(
                    "broadcast_dimensions contains invalid value "
                    f"{dim_index} for result with rank {result_rank}"
                )

            # (C5)  For all d in axes(operand):
            # dim(operand, d) = 1
            # or
            # dim(operand, d) = dim(result, broadcast_dimensions[d])
            if o_shape[i] != DYNAMIC_INDEX:
                dim_size = o_shape[i]
                result_dim_size = r_shape[dim_index]
                if dim_size not in (1, result_dim_size):
                    raise VerifyException(
                        f"size of operand dimension {i} ({dim_size}) "
                        "is not equal to 1 or size of result dimension "
                        f"{dim_index} ({result_dim_size})"
                    )


@irdl_op_definition
class ReshapeOp(IRDLOperation, ConditionallySpeculatableInterface):
    """
    Performs reshape of ``operand`` tensor to a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %result = stablehlo.reshape %operand : (tensor<2xf32>) -> tensor<1x2xf32>
    ```
    """

    name = "stablehlo.reshape"
    operand = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    assembly_format = """
    operands attr-dict `:` functional-type(operands, results)
    """

    traits = traits_def(
        NoMemoryEffect(),
        SameOperandsAndResultElementType(),
    )

    def is_speculatable(self) -> bool:
        operand_type = cast(TensorType, self.operand.type)
        return operand_type.has_static_shape()

    def verify_(self) -> None:
        """Verify that the operation has the same shape for all operands and results."""
        o_type = cast(TensorType, self.operands[0].type)
        r_type = self.result.type

        # Reshape requires a statically shaped result type.
        if not r_type.has_static_shape():
            raise VerifyException("reshape output must have a static shape.")

        # If the operand type is dynamically shaped there is nothing else to verify.
        if not o_type.has_static_shape():
            return

        # If the operand type is statically shaped (not required) the number of
        # elements must match that of the result type.
        num_operand_elements = 1
        for dim in o_type.get_shape():
            num_operand_elements *= dim

        num_result_elements = 1
        for dim in r_type.get_shape():
            num_result_elements *= dim

        if num_result_elements != num_operand_elements:
            raise VerifyException(
                "number of output elements ("
                f"{num_result_elements}"
                ") doesn't match expected number of elements ("
                f"{num_operand_elements}"
                ")"
            )


@irdl_op_definition
class SliceOp(IRDLOperation):
    """
    Extracts a slice from the ``operand`` using statically-computed starting
    indices and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = stablehlo.slice %operand [1:3, 4:8:2]
       : (tensor<3x8xi64>) -> tensor<2x2xi64>

    // Same in generic form: the `1:3` above is mapped to the first entry in
    // `start_indices` and `limit_indices`, while `strides` is implicitly 1.
    // The `4:8:2` above is parsed into the second entry of `start_indices`,
    // `limit_indices` and `strides` respectively.
    %result = "stablehlo.slice" (%operand) {
      start_indices = array<i64: 1, 4>,
      limit_indices = array<i64: 3, 8>,
      strides = array<i64: 1, 2>
    } : (tensor<3x8xi64>) -> tensor<2x2xi64>
    ```
    """

    name = "stablehlo.slice"

    operand = operand_def(AnyTensorType)
    start_indices = prop_def(DenseArrayBase.constr(i64))
    limit_indices = prop_def(DenseArrayBase.constr(i64))
    strides = prop_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)

    assembly_format = (
        "$operand custom<SliceRanges>($start_indices, $limit_indices, $strides)"
        "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (SliceRanges,)

    traits = traits_def(
        NoMemoryEffect(),
        SpeculatableIfStaticDimInOutputIsStaticInInput(),
        AllMatchSameOperatorTrait(
            ("start_indices", "limit_indices", "strides"), len, "size"
        ),
        SameOperandsAndResultElementType(),
    )
