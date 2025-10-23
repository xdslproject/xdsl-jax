# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
StableHLO attribute definitions for PennyLane's compiler infrastructure.

This module provides attribute definitions based on the StableHLO specification
(https://github.com/openxla/stablehlo/blob/main/docs/spec.md), including
attributes for StableHLO operations.
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import I64, ArrayAttr, IntegerAttr, i64
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer


# Utility functions for dimension array parsing/printing
def parse_dims(parser: AttrParser) -> ArrayAttr[IntegerAttr[I64]]:
    """Parse dimension array in [1, 2, 3] format"""
    value = parser.parse_comma_separated_list(
        AttrParser.Delimiter.SQUARE,
        lambda: IntegerAttr(parser.parse_integer(), i64),
    )
    return ArrayAttr(value)


def print_dims(printer: Printer, dims: ArrayAttr[IntegerAttr[I64]]):
    """Print dimension array in [1, 2, 3] format"""
    printer.print_string("[")
    printer.print_list(
        dims.data,
        lambda dim: printer.print_string(f"{dim.value.data}"),
    )
    printer.print_string("]")


class ComparisonDirection(StrEnum):
    """
    Comparison direction for stablehlo.
    """

    EQ = "EQ"
    NE = "NE"
    GE = "GE"
    GT = "GT"
    LE = "LE"
    LT = "LT"


@irdl_attr_definition
class ComparisonDirectionAttr(
    EnumAttribute[ComparisonDirection], SpacedOpaqueSyntaxAttribute
):
    """
    The values of `comparison_direction` and `compare_type` have the following
    semantics:
    For boolean and integer element types:

    * `EQ`: lhs = rhs.
    * `NE`: lhs != rhs.
    * `GE`: lhs >= rhs.
    * `GT`: lhs > rhs.
    * `LE`: lhs <= rhs
    * `LT`: lhs < rhs.

    For floating-point element types with `compare_type = FLOAT`, the op implements the
    following IEEE-754 operations:

    * `EQ`: compareQuietEqual.
    * `NE`: compareQuietNotEqual.
    * `GE`: compareQuietGreaterEqual.
    * `GT`: compareQuietGreater.
    * `LE`: compareQuietLessEqual.
    * `LT`: compareQuietLess.

    For floating-point element types with `compare_type = TOTALORDER`, the op uses the
    combination of totalOrder and compareQuietEqual operations from IEEE-754.
    For complex element types, lexicographic comparison of (real, imag) pairs is
    performed using the provided `comparison_direction` and `compare_type`.
    Imposing an ordering on complex numbers involves surprising semantics, so in the
    future we are planning to remove support for complex numbers when
    comparison_direction is `GE`, `GT`, `LE` or `LT`.

    For quantized types, performs `dequantize_compare(lhs, rhs, comparison_direction)`

    See external [documentation](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare).
    """

    name = "stablehlo.comparison_direction"


class ComparisonType(StrEnum):
    """
    Together with `comparison_direction` determines the semantics of comparison.

    See StableHLO's [source](https://github.com/openxla/stablehlo/blob/500b49459b2bd2f822644d4d9a43109e537d51e4/stablehlo/dialect/StablehloEnums.td#L152-L156).
    """

    NOTYPE = "NOTYPE"
    FLOAT = "FLOAT"
    TOTALORDER = "TOTALORDER"
    SIGNED = "SIGNED"
    UNSIGNED = "UNSIGNED"


@irdl_attr_definition
class ComparisonTypeAttr(EnumAttribute[ComparisonType], SpacedOpaqueSyntaxAttribute):
    """
    Together with `comparison_direction` determines the semantics of comparison.

    See external [documentation](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare).
    """

    name = "stablehlo.comparison_type"


class Precision(StrEnum):
    """
    XLA precision for an operand. Has backend specific meaning.
    """

    DEFAULT = "DEFAULT"
    HIGH = "HIGH"
    HIGHEST = "HIGHEST"


@irdl_attr_definition
class PrecisionAttr(EnumAttribute[Precision], SpacedOpaqueSyntaxAttribute):
    """
    XLA precision for an operand. Has backend specific meaning.

    See external [documentation](https://github.com/openxla/stablehlo/blob/b075e948092d8a27ed0be48f4f8dbaa6df7e2e3e/stablehlo/dialect/StablehloEnums.td#L46).
    """

    name = "stablehlo.precision"


@irdl_attr_definition
class TokenType(TypeAttribute, ParametrizedAttribute):
    """
    Token types represent tokens, i.e. opaque values produced and consumed by some
    operations.
    Tokens are used for imposing execution order on operations as described in the
    Execution section.

    E.g.,

    ```mlir
      // %input0: !stablehlo.token
      // %input1: !stablehlo.token
      %result = "stablehlo.after_all"(%input0, %input1)
              : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
    ```
    """

    name = "stablehlo.token"


@irdl_attr_definition
class DotAttr(ParametrizedAttribute):
    """
    Attribute that models the dimension information for dot.

    See external [documentation](https://github.com/openxla/stablehlo/blob/b075e948092d8a27ed0be48f4f8dbaa6df7e2e3e/stablehlo/dialect/StablehloAttrs.td#L82).
    """

    name = "stablehlo.dot"

    lhs_batching_dimensions: ArrayAttr[IntegerAttr[I64]]
    rhs_batching_dimensions: ArrayAttr[IntegerAttr[I64]]
    lhs_contracting_dimensions: ArrayAttr[IntegerAttr[I64]]
    rhs_contracting_dimensions: ArrayAttr[IntegerAttr[I64]]

    @staticmethod
    def _print_parameter(
        name: str, value: ArrayAttr[IntegerAttr[I64]], printer: Printer
    ):
        printer.print_string(f"\n{name} = [")
        printer.print_list(
            value.data,
            lambda dim: printer.print_string(f"{dim.value.data}"),
        )
        printer.print_string("]")

    @staticmethod
    def _parse_parameter(
        name: str, parser: AttrParser, optional: bool = False
    ) -> ArrayAttr[IntegerAttr[I64]]:
        if optional:
            if parser.parse_optional_characters(name) is None:
                return ArrayAttr(())
        else:
            parser.parse_characters(name)
        parser.parse_punctuation("=")
        value = parser.parse_comma_separated_list(
            AttrParser.Delimiter.SQUARE,
            lambda: IntegerAttr(parser.parse_integer(), i64),
        )
        return ArrayAttr(value)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            with printer.indented():
                if (
                    self.lhs_batching_dimensions.data
                    and self.rhs_batching_dimensions.data
                ):
                    DotAttr._print_parameter(
                        "lhs_batching_dimensions", self.lhs_batching_dimensions, printer
                    )
                    printer.print_string(",")
                    DotAttr._print_parameter(
                        "rhs_batching_dimensions", self.rhs_batching_dimensions, printer
                    )
                    printer.print_string(",")

                DotAttr._print_parameter(
                    "lhs_contracting_dimensions",
                    self.lhs_contracting_dimensions,
                    printer,
                )
                printer.print_string(",")
                DotAttr._print_parameter(
                    "rhs_contracting_dimensions",
                    self.rhs_contracting_dimensions,
                    printer,
                )
            printer.print_string("\n")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            lhs_batching_dimensions = DotAttr._parse_parameter(
                "lhs_batching_dimensions", parser, optional=True
            )
            if lhs_batching_dimensions.data:
                parser.parse_punctuation(",")
                rhs_batching_dimensions = DotAttr._parse_parameter(
                    "rhs_batching_dimensions", parser
                )
                parser.parse_punctuation(",")
            else:
                rhs_batching_dimensions = ArrayAttr(())

            lhs_contracting_dimensions = DotAttr._parse_parameter(
                "lhs_contracting_dimensions", parser
            )
            parser.parse_punctuation(",")
            rhs_contracting_dimensions = DotAttr._parse_parameter(
                "rhs_contracting_dimensions", parser
            )

            return (
                lhs_batching_dimensions,
                rhs_batching_dimensions,
                lhs_contracting_dimensions,
                rhs_contracting_dimensions,
            )


class ResultAccuracyMode(StrEnum):
    """
    XLA result accuracy mode.
    """

    DEFAULT = "DEFAULT"
    HIGH = "HIGHEST"
    HIGHEST = "TOLERANCE"


@irdl_attr_definition
class ResultAccuracyModeAttr(
    EnumAttribute[ResultAccuracyMode], SpacedOpaqueSyntaxAttribute
):
    """
    XLA result accuracy mode.

    See external [documentation](https://github.com/openxla/stablehlo/blob/7c50d4efeaea30bff6aa5e46c7f71170f5aa06af/stablehlo/dialect/StablehloEnums.td#L49-L70).
    """

    name = "stablehlo.result_accuracy_mode"


@irdl_attr_definition
class GatherDimensionNumbers(ParametrizedAttribute):
    """
    XLA gather dimension numbers.

    This attribute models the dimension information for gather operations.
    See external [documentation](https://github.com/openxla/stablehlo/blob/b075e948092d8a27ed0be48f4f8dbaa6df7e2e3e/stablehlo/dialect/StablehloAttrs.td#L42).
    """

    name = "stablehlo.gather"

    offset_dims: ArrayAttr[IntegerAttr[I64]]
    collapsed_slice_dims: ArrayAttr[IntegerAttr[I64]]
    operand_batching_dims: ArrayAttr[IntegerAttr[I64]]
    start_indices_batching_dims: ArrayAttr[IntegerAttr[I64]]
    start_index_map: ArrayAttr[IntegerAttr[I64]]
    index_vector_dim: IntegerAttr[I64]

    def print_parameters(self, printer: Printer) -> None:
        """Print gather dimension numbers in structured format"""
        with printer.in_angle_brackets():
            with printer.indented():
                # Print offset_dims
                printer.print_string("\noffset_dims = ")
                print_dims(printer, self.offset_dims)
                printer.print_string(",")

                # Print collapsed_slice_dims
                printer.print_string("\ncollapsed_slice_dims = ")
                print_dims(printer, self.collapsed_slice_dims)
                printer.print_string(",")

                # Print operand_batching_dims
                printer.print_string("\noperand_batching_dims = ")
                print_dims(printer, self.operand_batching_dims)
                printer.print_string(",")

                # Print start_indices_batching_dims
                printer.print_string("\nstart_indices_batching_dims = ")
                print_dims(printer, self.start_indices_batching_dims)
                printer.print_string(",")

                # Print start_index_map
                printer.print_string("\nstart_index_map = ")
                print_dims(printer, self.start_index_map)
                printer.print_string(",")

                # Print index_vector_dim
                printer.print_string(
                    f"\nindex_vector_dim = {self.index_vector_dim.value.data}"
                )
            printer.print_string("\n")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse gather dimension numbers from structured format"""
        with parser.in_angle_brackets():
            # Initialize default values for all fields
            offset_dims = ArrayAttr([])
            collapsed_slice_dims = ArrayAttr([])
            operand_batching_dims = ArrayAttr([])
            start_indices_batching_dims = ArrayAttr([])
            start_index_map = ArrayAttr([])
            index_vector_dim = IntegerAttr(0, i64)

            # Try to parse offset_dims
            if parser.parse_optional_characters("offset_dims") is not None:
                parser.parse_punctuation("=")
                offset_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse collapsed_slice_dims
            if parser.parse_optional_characters("collapsed_slice_dims") is not None:
                parser.parse_punctuation("=")
                collapsed_slice_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse operand_batching_dims
            if parser.parse_optional_characters("operand_batching_dims") is not None:
                parser.parse_punctuation("=")
                operand_batching_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse start_indices_batching_dims
            if (
                parser.parse_optional_characters("start_indices_batching_dims")
                is not None
            ):
                parser.parse_punctuation("=")
                start_indices_batching_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse start_index_map
            if parser.parse_optional_characters("start_index_map") is not None:
                parser.parse_punctuation("=")
                start_index_map = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse index_vector_dim
            if parser.parse_optional_characters("index_vector_dim") is not None:
                parser.parse_punctuation("=")
                index_vector_dim = IntegerAttr(parser.parse_integer(), i64)

            return (
                offset_dims,
                collapsed_slice_dims,
                operand_batching_dims,
                start_indices_batching_dims,
                start_index_map,
                index_vector_dim,
            )


@irdl_attr_definition
class ScatterDimensionNumbers(ParametrizedAttribute):
    """
    XLA scatter dimension numbers.

    This attribute models the dimension information for scatter operations.
    See external [documentation](https://github.com/openxla/stablehlo/blob/b075e948092d8a27ed0be48f4f8dbaa6df7e2e3e/stablehlo/dialect/StablehloAttrs.td#L28).
    """

    name = "stablehlo.scatter"

    update_window_dims: ArrayAttr[IntegerAttr[I64]]
    inserted_window_dims: ArrayAttr[IntegerAttr[I64]]
    input_batching_dims: ArrayAttr[IntegerAttr[I64]]
    scatter_indices_batching_dims: ArrayAttr[IntegerAttr[I64]]
    scatter_dims_to_operand_dims: ArrayAttr[IntegerAttr[I64]]
    index_vector_dim: IntegerAttr[I64]

    def print_parameters(self, printer: Printer) -> None:
        """Print scatter dimension numbers in structured format"""
        with printer.in_angle_brackets():
            with printer.indented():
                # Print update_window_dims
                printer.print_string("\nupdate_window_dims = ")
                print_dims(printer, self.update_window_dims)
                printer.print_string(",")

                # Print inserted_window_dims
                printer.print_string("\ninserted_window_dims = ")
                print_dims(printer, self.inserted_window_dims)
                printer.print_string(",")

                # Print input_batching_dims
                printer.print_string("\ninput_batching_dims = ")
                print_dims(printer, self.input_batching_dims)
                printer.print_string(",")

                # Print scatter_indices_batching_dims
                printer.print_string("\nscatter_indices_batching_dims = ")
                print_dims(printer, self.scatter_indices_batching_dims)
                printer.print_string(",")

                # Print scatter_dims_to_operand_dims
                printer.print_string("\nscatter_dims_to_operand_dims = ")
                print_dims(printer, self.scatter_dims_to_operand_dims)
                printer.print_string(",")

                # Print index_vector_dim
                printer.print_string(
                    f"\nindex_vector_dim = {self.index_vector_dim.value.data}"
                )
            printer.print_string("\n")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        """Parse scatter dimension numbers from structured format"""
        with parser.in_angle_brackets():
            # Initialize default values for all fields
            update_window_dims = ArrayAttr([])
            inserted_window_dims = ArrayAttr([])
            input_batching_dims = ArrayAttr([])
            scatter_indices_batching_dims = ArrayAttr([])
            scatter_dims_to_operand_dims = ArrayAttr([])
            index_vector_dim = IntegerAttr(0, i64)

            # Try to parse update_window_dims
            if parser.parse_optional_characters("update_window_dims") is not None:
                parser.parse_punctuation("=")
                update_window_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse inserted_window_dims
            if parser.parse_optional_characters("inserted_window_dims") is not None:
                parser.parse_punctuation("=")
                inserted_window_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse input_batching_dims
            if parser.parse_optional_characters("input_batching_dims") is not None:
                parser.parse_punctuation("=")
                input_batching_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse scatter_indices_batching_dims
            if (
                parser.parse_optional_characters("scatter_indices_batching_dims")
                is not None
            ):
                parser.parse_punctuation("=")
                scatter_indices_batching_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse scatter_dims_to_operand_dims
            if (
                parser.parse_optional_characters("scatter_dims_to_operand_dims")
                is not None
            ):
                parser.parse_punctuation("=")
                scatter_dims_to_operand_dims = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            # Try to parse index_vector_dim
            if parser.parse_optional_characters("index_vector_dim") is not None:
                parser.parse_punctuation("=")
                index_vector_dim = IntegerAttr(parser.parse_integer(), i64)

            return (
                update_window_dims,
                inserted_window_dims,
                input_batching_dims,
                scatter_indices_batching_dims,
                scatter_dims_to_operand_dims,
                index_vector_dim,
            )


class CustomCallApiVersion(StrEnum):
    """StableHLO CustomCall API version."""

    API_VERSION_UNSPECIFIED = "API_VERSION_UNSPECIFIED"
    API_VERSION_ORIGINAL = "API_VERSION_ORIGINAL"
    API_VERSION_STATUS_RETURNING = "API_VERSION_STATUS_RETURNING"
    API_VERSION_STATUS_RETURNING_UNIFIED = "API_VERSION_STATUS_RETURNING_UNIFIED"
    API_VERSION_TYPED_FFI = "API_VERSION_TYPED_FFI"


@irdl_attr_definition
class CustomCallApiVersionAttr(
    EnumAttribute[CustomCallApiVersion], SpacedOpaqueSyntaxAttribute
):
    """StableHLO custom call API version attribute.

    Mirrors StableHLO enum for CustomCall API versions.
    """

    name = "stablehlo.custom_call_api_version"


@irdl_attr_definition
class OutputOperandAlias(ParametrizedAttribute):
    """
    This attribute captures the alias relationship of the output to one of the
    operands for a ``CustomCall`` op, denoted by ``operand_index``. The
    ``output_tuple_indices`` and ``operand_tuple_indices`` are used to index into
    output and operand types. These indices lists are empty if the corresponding
    types are not tuple types, and can be arbitrarily long in case of
    arbitrarily nested tuple types.

    See https://www.tensorflow.org/xla/aliasing.

    Example when used as array with in stablehlo.custom-call:

    ```mlir
    %0 = "stablehlo.custom_call"(%arg0, %arg1) {
      // other attributes
      output_operand_alias = [
        #stablehlo.output_operand_alias<output_tuple_indices = [0],
                                   operand_index = 0,
                                   operand_tuple_indices = [1]>
      ]
    } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>)
      -> tuple<tensor<2x3xf32>>

    The output and the 0th operand are both tuples. The aliasing shows the
    relationship between the 0th element in output tuple with the 1st element in
    the 0th operand. And both of them are of the same type: ``tensor<2x3xf32>``.
    ```
    """

    name = "stablehlo.output_operand_alias"

    output_tuple_indices: ArrayAttr[IntegerAttr[I64]]
    operand_index: IntegerAttr[I64]
    operand_tuple_indices: ArrayAttr[IntegerAttr[I64]]

    def print_parameters(self, printer: Printer) -> None:
        """Print the OutputOperandAlias attribute."""
        with printer.in_angle_brackets():
            with printer.indented():
                printer.print_string("\noutput_tuple_indices = ")
                print_dims(printer, self.output_tuple_indices)
                printer.print_string(",")

                printer.print_string("\noperand_index = ")
                printer.print_string(f"{self.operand_index.value.data}")
                printer.print_string(",")

                printer.print_string("\noperand_tuple_indices = ")
                print_dims(printer, self.operand_tuple_indices)
            printer.print_string("\n")

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        """Parse the OutputOperandAlias attribute."""
        with parser.in_angle_brackets():
            output_tuple_indices = ArrayAttr([])
            operand_index = IntegerAttr(0, i64)
            operand_tuple_indices = ArrayAttr([])

            if parser.parse_optional_characters("output_tuple_indices") is not None:
                parser.parse_punctuation("=")
                output_tuple_indices = parse_dims(parser)
                parser.parse_optional_punctuation(",")

            if parser.parse_optional_characters("operand_index") is not None:
                parser.parse_punctuation("=")
                operand_index = IntegerAttr(parser.parse_integer(), i64)
                parser.parse_optional_punctuation(",")

            if parser.parse_optional_characters("operand_tuple_indices") is not None:
                parser.parse_punctuation("=")
                operand_tuple_indices = parse_dims(parser)

            return (output_tuple_indices, operand_index, operand_tuple_indices)
