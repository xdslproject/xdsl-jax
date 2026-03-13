"""
Dynamism operations for the StableHLO dialect.
"""

from typing import cast

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnyTensorType,
    DenseArrayBase,
    TensorType,
    i64,
)
from xdsl.interfaces import ConditionallySpeculatableInterface
from xdsl.ir import Attribute, Operation
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import (
    ConditionallySpeculatable,
    ConstantLike,
    NoMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException

from .types import DimensionTensorType


@irdl_op_definition
class DynamicBroadcastInDimOp(IRDLOperation, ConditionallySpeculatableInterface):
    """
    This operation is functionally identical to
    [broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim)
    op, but the result shape is specified dynamically via ``output_dimensions``.

    It also accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_broadcast_in_dim

    Example:
    ```mlir
    %operand = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
    %output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
    %result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
      broadcast_dimensions = array<i64: 2, 1>,
      known_expanding_dimensions = array<i64: 0>,
      known_nonexpanding_dimensions = array<i64: 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    ```
    """

    name = "stablehlo.dynamic_broadcast_in_dim"

    operand = operand_def(AnyTensorType)
    output_dimensions = operand_def(DimensionTensorType)
    broadcast_dimensions = prop_def(DenseArrayBase.constr(i64))
    known_expanding_dimensions = opt_prop_def(DenseArrayBase.constr(i64))
    known_nonexpanding_dimensions = opt_prop_def(DenseArrayBase.constr(i64))
    result = result_def(AnyTensorType)

    assembly_format = (
        "$operand `,` $output_dimensions `,` `dims` `=` $broadcast_dimensions "
        "attr-dict `:` functional-type(operands, results)"
    )

    traits = traits_def(
        ConditionallySpeculatable(),
        NoMemoryEffect(),
    )

    irdl_options = (ParsePropInAttrDict(),)

    def is_speculatable(self) -> bool:
        """Check if the operation is speculatable."""
        operand_ty = cast(TensorType[Attribute], self.operand.type)
        # If input is dynamic, the broadcasting rules might be violated at runtime,
        # so not speculatable.
        if not operand_ty.has_static_shape():
            return False
        # If input is broadcastable (all 1's) and result is fully dynamic, speculatable.
        result_ty = self.result.type
        result_dynamic = all(dim == DYNAMIC_INDEX for dim in result_ty.get_shape())
        if operand_ty.element_count() == 1 and result_dynamic:
            return True

        # If shape is known, speculatable.
        output_owner = self.output_dimensions.owner
        if isinstance(output_owner, Operation) and output_owner.has_trait(ConstantLike):
            return True

        return False

    def _verify_rank_constraints(
        self,
        bcast_dims: tuple[int, ...],
        operand_ty: TensorType[Attribute],
        result_ty: TensorType[Attribute],
    ) -> None:
        """Verify then operand and result tensors against the rank constraints."""

        operand_rank = operand_ty.get_num_dims()
        result_rank = result_ty.get_num_dims()

        # (c2) broadcast_dimensions size == operand rank
        if len(bcast_dims) != operand_rank:
            raise VerifyException(
                "broadcast_dimensions size ("
                f"{len(bcast_dims)}"
                ") does not match operand rank ("
                f"{operand_rank}"
                ")"
            )

        # (c3) result rank >= operand rank
        if result_rank < operand_rank:
            raise VerifyException(
                "result rank ("
                f"{result_rank}"
                ") is less than operand rank ("
                f"{operand_rank}"
                ")"
            )

        # (c7) output_dimensions shape compatible with result rank
        out_dims_ty = self.output_dimensions.type  # pylint: disable=no-member
        assert isinstance(out_dims_ty, TensorType)
        # Must be rank-1 tensor, and length must match result rank when statically known
        out_shape = out_dims_ty.get_shape()
        if len(out_shape) != 1:
            raise VerifyException("output_dimensions must be a 1D tensor")
        if out_shape[0] != -1 and out_shape[0] != result_rank:
            raise VerifyException(
                "length of output_dimensions ("
                f"{out_shape[0]}"
                ") is not compatible with result rank ("
                f"{result_rank}"
                ")"
            )

    def _verify_per_dimension_bounds(
        self,
        bcast_dims: tuple[int, ...],
        operand_ty: TensorType[Attribute],
        result_ty: TensorType[Attribute],
    ) -> None:
        """Verify compatibility of operand and result dimensions."""
        # (c5) bounds and per-dimension compatibility
        operand_shape = operand_ty.get_shape()
        result_shape = result_ty.get_shape()
        result_rank = result_ty.get_num_dims()

        for i, dim_index in enumerate(bcast_dims):
            if dim_index < 0 or dim_index >= result_rank:
                raise VerifyException(
                    "broadcast_dimensions contains invalid value "
                    f"{dim_index} for result with rank {result_rank}"
                )
            op_dim = operand_shape[i]
            res_dim = result_shape[dim_index]
            # If operand dim is static and not size-1,
            # require compatibility with result dim
            if op_dim not in (-1, 1):
                if res_dim not in (-1, op_dim):
                    raise VerifyException(
                        "size of operand dimension "
                        f"{i} ({op_dim}) is not compatible with size of "
                        f"result dimension {dim_index} ({res_dim})"
                    )

    def _verify_expansion_hints(self, operand_ty: TensorType[Attribute]) -> None:
        """Verify the operation's expansion hints."""
        # (c8) no duplicate expansion hints across both lists
        operand_rank = operand_ty.get_num_dims()

        hints: list[int] = []
        if self.known_expanding_dimensions is not None:
            hints.extend(self.known_expanding_dimensions.get_values())
        if self.known_nonexpanding_dimensions is not None:
            hints.extend(self.known_nonexpanding_dimensions.get_values())
        if len(set(hints)) != len(hints):
            raise VerifyException(
                "duplicate expansion hint for at least one operand dimension"
            )

        # (c9/c10) each hint must reference a valid operand dimension
        for h in set(hints):
            if h < 0 or h >= operand_rank:
                raise VerifyException(
                    "hint for expanding dimension "
                    f"{h} does not refer to a valid operand dimension"
                )

    def verify_(self) -> None:
        """Verify the operation."""
        operand_ty = cast(TensorType[Attribute], self.operand.type)
        result_ty = self.result.type
        bcast_dims = tuple(self.broadcast_dimensions.get_values())

        self._verify_rank_constraints(bcast_dims, operand_ty, result_ty)

        # (c4) broadcast_dimensions should not have duplicates
        if len(set(bcast_dims)) != len(bcast_dims):
            raise VerifyException("broadcast_dimensions should not have duplicates")

        self._verify_per_dimension_bounds(bcast_dims, operand_ty, result_ty)

        self._verify_expansion_hints(operand_ty)
