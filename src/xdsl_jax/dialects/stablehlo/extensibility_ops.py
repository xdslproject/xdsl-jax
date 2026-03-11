"""
StableHLO extensibility operations (e.g. custom_call) for implementation-defined
or backend-specific behavior.
"""

from typing import cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    DenseIntElementsAttr,
    DictionaryAttr,
    FlatSymbolRefAttr,
    IntegerAttr,
    StringAttr,
    TensorType,
    TupleType,
    i32,
)
from xdsl.ir import Attribute, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import MemoryEffect
from xdsl.utils.exceptions import VerifyException

from .attributes import CustomCallApiVersion, OutputOperandAlias
from .custom_directives import CustomCallTarget

_CUSTOM_CALL_API_VERSION_VALUES = {
    CustomCallApiVersion.API_VERSION_UNSPECIFIED: 0,
    CustomCallApiVersion.API_VERSION_ORIGINAL: 1,
    CustomCallApiVersion.API_VERSION_STATUS_RETURNING: 2,
    CustomCallApiVersion.API_VERSION_STATUS_RETURNING_UNIFIED: 3,
    CustomCallApiVersion.API_VERSION_TYPED_FFI: 4,
}
_CUSTOM_CALL_API_VERSION_NAMES = {
    value: enum.value for enum, value in _CUSTOM_CALL_API_VERSION_VALUES.items()
}


def _resolve_tuple_element_type(
    base: Attribute, indices: list[int], attr_name: str
) -> Attribute:
    """
    Resolve the element type of a (possibly nested) tuple type by following
    the given indices. Used when validating output_operand_alias indices.
    Raises VerifyException if indices are out of bounds or base is not a tuple.
    """
    part = base
    for idx in indices:
        if not isinstance(part, TupleType):
            raise VerifyException(
                f"{attr_name} in the output_operand_alias attribute out of bounds"
            )
        tuple_types = part.types.data
        if idx < 0 or idx >= len(tuple_types):
            raise VerifyException(
                f"{attr_name} in the output_operand_alias attribute out of bounds"
            )
        part = tuple_types[idx]
    return part


def _check_one_type_layout(
    ty: Attribute,
    layout_attr: Attribute,
    index: int,
    value_name: str,
) -> None:
    """
    Verify that a single type has a valid layout: no tuple types, non-tensors
    must have empty layout, tensors must have a permutation of [0, rank).
    """
    if isinstance(ty, TupleType):
        raise VerifyException(
            "Tuple types are not fully supported with layout constraints yet"
        )
    layout = list(cast(DenseIntElementsAttr, layout_attr).get_values())
    if not isinstance(ty, TensorType):
        if len(layout) == 0:
            return
        raise VerifyException(
            "Only tensor types can have non-empty layout: "
            f"{value_name} #{index} of type {ty} has layout {layout}"
        )
    rank = ty.get_num_dims()
    if rank != len(layout) or sorted(layout) != list(range(rank)):
        raise VerifyException(
            f"incorrect layout {layout} for type {ty}, layout must be a permutation "
            f"of [0, {rank})"
        )


def _check_types_vs_layouts(
    types: tuple[Attribute, ...],
    layouts: ArrayAttr,
    value_name: str,
) -> None:
    """Verify that types and layouts counts match and each type has a valid layout."""
    if len(types) != len(layouts.data):
        raise VerifyException(
            "Number of "
            f"{value_name}s must match the number of {value_name} layouts, "
            f"{len(types)} != {len(layouts.data)}"
        )
    for index, (ty, layout_attr) in enumerate(zip(types, layouts.data)):
        _check_one_type_layout(ty, layout_attr, index, value_name)


@irdl_op_definition
class CustomCallOp(IRDLOperation):
    """
    Encapsulates an implementation-defined operation ``call_target_name`` that
    takes ``inputs`` and ``called_computations`` and produces ``results``.

    Depending on the API version there are two ways to pass extra bits of static
    information to the external function:
    1. Use ``API_VERSION_TYPED_FFI`` which allows passing a dictionary attribute.
    2. Use a previous API version with a ``StringAttr`` to encode backend config.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call

    Example:
    ```mlir
    %results = stablehlo.custom_call @foo(%input0) {
      backend_config = {bar = 42 : i32},
      api_version = 4 : i32,
      called_computations = [@foo]
    } : (tensor<f64>) -> tensor<f64>
    ```
    """

    name = "stablehlo.custom_call"

    inputs = var_operand_def(AnyAttr())
    call_target_name = prop_def(StringAttr)
    has_side_effect = prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))
    backend_config = opt_prop_def(DictionaryAttr | StringAttr)
    api_version = prop_def(
        IntegerAttr,
        default_value=IntegerAttr(
            _CUSTOM_CALL_API_VERSION_VALUES[CustomCallApiVersion.API_VERSION_ORIGINAL],
            i32,
        ),
    )
    called_computations = opt_prop_def(
        ArrayAttr[FlatSymbolRefAttr], default_value=ArrayAttr([])
    )
    operand_layouts = opt_prop_def(ArrayAttr[DenseIntElementsAttr])
    result_layouts = opt_prop_def(ArrayAttr[DenseIntElementsAttr])
    output_operand_aliases = prop_def(
        ArrayAttr[OutputOperandAlias], default_value=ArrayAttr([])
    )

    result = var_result_def(AnyAttr())

    traits = traits_def(
        MemoryEffect(),
    )

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = (
        "custom<CustomCallTarget>($call_target_name) `(` $inputs `)`"
        "attr-dict `:` functional-type(operands, results)"
    )

    custom_directives = (CustomCallTarget,)

    def verify_(self) -> None:
        """Verify the CustomCallOp."""
        self._verify_layout_attributes()
        self._verify_output_operand_aliases()
        self._verify_api_version_and_backend_config()

    def _verify_layout_attributes(self) -> None:
        """
        Verify that operand_layouts and result_layouts are either both set or
        both unset. When set, verify that each operand/result type has a
        matching layout (same count, valid permutation for tensors, empty for
        non-tensors like token). Tuple types are not supported with layouts yet.
        """
        operand_layouts = self.operand_layouts
        result_layouts = self.result_layouts
        if (operand_layouts is None) != (result_layouts is None):
            raise VerifyException(
                "Layout attributes should be specified for either both operands and "
                "results or none."
            )
        if operand_layouts is None:
            return
        assert result_layouts is not None

        operand_types = tuple(op.type for op in self.operands)
        if len(self.result_types) == 1 and isinstance(self.result_types[0], TupleType):
            result_types = tuple(self.result_types[0].types.data)
        else:
            result_types = tuple(self.result_types)
        _check_types_vs_layouts(operand_types, operand_layouts, "operand")
        _check_types_vs_layouts(result_types, result_layouts, "result")

    def _verify_output_operand_aliases(self) -> None:
        """
        Verify each output_operand_alias: operand_index in range, and that
        the resolved operand type (via operand_tuple_indices) matches the
        resolved output type (via output_tuple_indices).
        """
        if len(self.result_types) > 1:
            output_base: Attribute = TupleType(
                cast(tuple[TypeAttribute, ...], self.result_types)
            )
        else:
            output_base = self.result_types[0]

        for alias in self.output_operand_aliases.data:
            output_tuple_indices = [
                idx.value.data for idx in alias.output_tuple_indices.data
            ]
            operand_index = alias.operand_index.value.data
            operand_tuple_indices = [
                idx.value.data for idx in alias.operand_tuple_indices.data
            ]

            if operand_index < 0 or operand_index >= len(self.operands):
                raise VerifyException(
                    "expects operandIndex in the output_operand_alias attribute "
                    "to be in range [0, "
                    f"{len(self.operands)}); got: {operand_index}."
                )

            operand_part = _resolve_tuple_element_type(
                self.operands[operand_index].type,
                operand_tuple_indices,
                "operand_tuple_indices",
            )
            output_part = _resolve_tuple_element_type(
                output_base,
                output_tuple_indices,
                "output_tuple_indices",
            )
            if operand_part != output_part:
                raise VerifyException(
                    "shapes mismatch in the output_operand_alias attribute: "
                    f"operand part has type {operand_part} and "
                    f"output part has type {output_part}"
                )

    def _verify_api_version_and_backend_config(self) -> None:
        """
        Verify api_version is a known value, and when backend_config is set,
        that it is a dictionary for API_VERSION_TYPED_FFI (4) or a string
        for other API versions.
        """
        api_version_value = self.api_version.value.data
        api_version_name = _CUSTOM_CALL_API_VERSION_NAMES.get(
            api_version_value, str(api_version_value)
        )
        if api_version_value not in _CUSTOM_CALL_API_VERSION_NAMES:
            raise VerifyException(f"invalid api_version value {api_version_value}")

        if self.backend_config is None:
            return
        if (
            api_version_value
            == _CUSTOM_CALL_API_VERSION_VALUES[
                CustomCallApiVersion.API_VERSION_TYPED_FFI
            ]
        ):
            if not isinstance(self.backend_config, DictionaryAttr):
                raise VerifyException(
                    "backend_config for api_version "
                    f"{api_version_name} must be a dictionary attribute."
                )
        else:
            if not isinstance(self.backend_config, StringAttr):
                raise VerifyException(
                    "backend_config for api_version "
                    f"{api_version_name} must be a string attribute."
                )
