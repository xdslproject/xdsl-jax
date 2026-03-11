"""
Data movement operations for the StableHLO dialect.
"""

from xdsl.dialects.builtin import (
<<<<<<< add-scatter-gather-ops
    AnyTensorType,
    BoolAttr,
=======
    DYNAMIC_INDEX,
    AnyTensorType,
>>>>>>> main
    DenseArrayBase,
    TensorType,
    i64,
)
from xdsl.irdl import (
    IRDLOperation,
    SameVariadicOperandSize,
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
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
)
<<<<<<< add-scatter-gather-ops
from xdsl.utils.type import get_element_type_or_self
=======
from xdsl.utils.exceptions import VerifyException
>>>>>>> main

from xdsl_jax.xdsl_extras import (
    AllMatchSameOperatorTrait,
    SameOperandsAndResultElementType,
)

from .attributes import GatherDimensionNumbers, ScatterDimensionNumbers
from .custom_directives import SliceRanges
from .traits import (
<<<<<<< add-scatter-gather-ops
    SpeculatableIfStaticDimInOutputIsStaticInInput,
)
from .types import IntegerOrIndexTensorType, IntegerTensorType


@irdl_op_definition
class GatherOp(IRDLOperation):
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
class ScatterOp(IRDLOperation):
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
=======
    SpeculatableIfAllInputsStatic,
    SpeculatableIfStaticDimInOutputIsStaticInInput,
)


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
>>>>>>> main


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
        ConditionallySpeculatable(),
        SpeculatableIfStaticDimInOutputIsStaticInInput(),
        AllMatchSameOperatorTrait(
            ("start_indices", "limit_indices", "strides"), len, "size"
        ),
        SameOperandsAndResultElementType(),
    )
