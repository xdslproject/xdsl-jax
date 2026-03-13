"""
Custom directives for the StableHLO dialect.
"""

import re
from typing import Any, cast

from xdsl.dialects.builtin import (
    I64,
    ArrayAttr,
    BoolAttr,
    ComplexType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    TensorType,
    i32,
    i64,
)
from xdsl.ir import Attribute
from xdsl.irdl import IRDLOperation
from xdsl.irdl.declarative_assembly_format import (
    AttributeVariable,
    CustomDirective,
    FunctionalTypeDirective,
    ParsingState,
    PrintingState,
    TypeDirective,
    VariableDirective,
    irdl_custom_directive,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.hints import isa

from .attributes import DotAlgorithmAttr, DotAttr, Precision, PrecisionAttr


def _create_real_type(shaped_type: TensorType[Attribute]) -> TensorType[Attribute]:
    """
    Takes a tensor type that may have complex elements and returns a type
    with the same shape but real numeric element type.

    Ex: tensor<4xcomplex<f32>> -> tensor<4xf32>
    """
    element_type: Attribute = shaped_type.element_type
    if isa(element_type, ComplexType):
        element_type = element_type.element_type
    return TensorType(element_type, shaped_type.get_shape(), shaped_type.encoding)


@irdl_custom_directive
class SameOperandsAndResultType(CustomDirective):
    """
    Custom directive that prints/parses types when operands and result have the
    same type.

    - Print: If all operand types and result type are the same, print just the type
      once. Otherwise, print functional type `(operand_types) -> result_type`.
    - Parse: Parse a type. If it's a function type, use those types for operands
      and result. Otherwise, the single type applies to all operands and the result.
    """

    operand_types: TypeDirective
    result_type: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        # Try to parse a function type first
        functional_type = FunctionalTypeDirective(
            self.operand_types.inner, self.result_type.inner
        )
        if functional_type.parse(parser, state):
            return True

        # Single type: applies to all operands and result
        single_type = parser.parse_type()
        inner = self.operand_types.inner
        if isinstance(inner, VariableDirective):
            operands = state.operands[inner.index]
            n_operands = len(operands) if operands else 0
        else:
            n_operands = len(state.operand_types)
        self.operand_types.set(state, (single_type,) * n_operands)
        self.result_type.set(state, (single_type,))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operand_types = self.operand_types.get(op)
        result_types = self.result_type.get(op)

        state.print_whitespace(printer)

        # `() -> a` or `(a, a, ...) -> a` prints `a`
        result_type = result_types[0]
        if all(t == result_type for t in operand_types):
            printer.print_attribute(result_type)
            return

        # Fall back to generic
        printer.print_function_type(operand_types, result_types)


@irdl_custom_directive
class ComplexOpType(CustomDirective):
    """
    Custom directive that prints/parses types for the stablehlo.complex op.

    The complex op takes two real-typed operands (lhs, rhs) and produces a
    complex-typed result. The operand types are inferred from the result type.

    - Print: If both operand types equal the "real type" derived from the result
      (same shape, element type stripped of complex wrapper), print just the
      result type. Otherwise, fall back to functional type.
    - Parse: Parse a type. If it's a function type, assign types directly.
      Otherwise, expect a tensor with complex element type, and infer the
      operand types from the real component of the complex element type.
    """

    operand_types: TypeDirective
    result_types: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        # Handle function type fallback: (lhs_type, rhs_type) -> result_type
        functional_type = FunctionalTypeDirective(
            self.operand_types.inner, self.result_types.inner
        )
        if functional_type.parse(parser, state):
            return True

        # Single type: operand type is inferred from complex result type
        parsed_type = parser.parse_type()
        if not isa(parsed_type, TensorType[ComplexType]):
            parser.raise_error("expected tensor with complex element type")

        real_type = _create_real_type(parsed_type)
        self.operand_types.set(state, (real_type, real_type))
        self.result_types.set(state, (parsed_type,))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operand_types = self.operand_types.get(op)
        result_types = self.result_types.get(op)
        result_type = cast(TensorType[Attribute], result_types[0])

        state.print_whitespace(printer)

        real_type = _create_real_type(result_type)

        if all(t == real_type for t in operand_types):
            printer.print_attribute(result_type)
            return

        # Fall back to functional type
        printer.print_function_type(operand_types, result_types)


@irdl_custom_directive
class ConstantOpValue(CustomDirective):
    """
    Custom directive for stablehlo.constant that prints and parses the value
    attribute in stripped form, then infers the result type from the attribute's type.
    """

    value: AttributeVariable
    result_type: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        attr = cast(DenseIntOrFPElementsAttr, parser.parse_attribute())
        self.value.set(state, attr)
        self.result_type.set(state, (attr.type,))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        attr = cast(DenseIntOrFPElementsAttr, self.value.get(op))
        state.print_whitespace(printer)
        attr.print_without_type(printer)
        printer.print_string(" : ")
        printer.print_attribute(attr.type)


@irdl_custom_directive
class CustomCallTarget(CustomDirective):
    """
    Custom directive for stablehlo.custom_call call_target_name.

    Prints/parses the target as a symbol name (e.g., `@foo`).
    """

    call_target_name: AttributeVariable

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        target = parser.parse_symbol_name()
        self.call_target_name.set(state, target)
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        target = cast(StringAttr, self.call_target_name.get(op))
        state.print_whitespace(printer)
        printer.print_symbol_name(target.data)


@irdl_custom_directive
class PairwiseOpType(CustomDirective):
    """
    Custom directive for ops with pairwise same operand and result types
    (e.g. stablehlo.optimization_barrier).

    Mirrors MLIR StableHLO's printPairwiseOpType/parsePairwiseOpType:
    - Parse: one type list (e.g. `tensor<f32>, tensor<f32>`) is parsed and used
      for both operands and results (results = operands).
    - Print: operand types are printed comma-separated (result types are the same).
    """

    operand_types: TypeDirective
    result_types: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_type
        )
        self.operand_types.set(state, types)
        self.result_types.set(state, types)
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operand_types = self.operand_types.get(op)
        state.print_whitespace(printer)
        printer.print_list(operand_types, printer.print_attribute)


@irdl_custom_directive
class SelectOpType(CustomDirective):
    """
    Custom directive for stablehlo.select printing/parsing.

    - Print: If on_true/on_false types match the result type, print `pred, result`.
      Otherwise print a functional type `(pred, on_true, on_false) -> result`.
    - Parse: Parse either a list of two types (`pred_type, op_and_result_type`)
      or a functional type describing all operand/result types.
    """

    operand_types: TypeDirective
    result_types: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        functional_type = FunctionalTypeDirective(
            self.operand_types.inner, self.result_types.inner
        )
        if functional_type.parse(parser, state):
            return True

        types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_type
        )

        if len(types) == 2:
            pred_type, op_result_type = types
            self.operand_types.set(state, (pred_type, op_result_type, op_result_type))
            self.result_types.set(state, (op_result_type,))
            return True

        parser.raise_error("expected functional type or list of two types")

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operand_types = self.operand_types.get(op)
        result_types = self.result_types.get(op)

        on_true_type, on_false_type = operand_types[1:]
        result_type = result_types[0]

        state.print_whitespace(printer)

        if on_true_type != result_type or on_false_type != result_type:
            printer.print_function_type(operand_types, result_types)
            return

        printer.print_attribute(operand_types[0])
        printer.print_string(", ")
        printer.print_attribute(result_type)


@irdl_custom_directive
class ExponentMantissa(CustomDirective):
    """
    Custom directive for stablehlo.reduce_precision that prints and parses the
    exponent and mantissa attributes.
    """

    exponent: AttributeVariable
    mantissa: AttributeVariable

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        exp_man = parser.parse_identifier()
        if not (match := re.fullmatch(r"e([0-9]+)m([0-9]+)", exp_man)):
            parser.raise_error(
                f"expected exponent mantissa in format e#m#, saw {exp_man}"
            )

        exponent, mantissa = map(int, match.groups())
        self.exponent.set(state, IntegerAttr(exponent, i32))
        self.mantissa.set(state, IntegerAttr(mantissa, i32))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        exponent_attr = cast(IntegerAttr, self.exponent.get(op))
        mantissa_attr = cast(IntegerAttr, self.mantissa.get(op))

        state.print_whitespace(printer)
        printer.print_string(
            f"e{exponent_attr.value.data:d}m{mantissa_attr.value.data:d}"
        )


@irdl_custom_directive
class SliceRanges(CustomDirective):
    """
    Custom directive for stablehlo.slice that prints and parses the start indices,
    limit indices and strides attributes.

    Format: `[start:limit[:stride], ...]` — stride defaults to 1 if omitted.
    """

    start_indices: AttributeVariable
    limit_indices: AttributeVariable
    strides: AttributeVariable

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        def parse_range() -> tuple[int, int, int]:
            start = parser.parse_integer()
            parser.parse_punctuation(":")
            limit = parser.parse_integer()
            if parser.parse_optional_punctuation(":") is not None:
                stride = parser.parse_integer()
            else:
                stride = 1
            return start, limit, stride

        ranges = parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parse_range)
        start, limit, stride = zip(*ranges) if ranges else ((), (), ())
        self.start_indices.set(state, DenseArrayBase.from_list(i64, start))
        self.limit_indices.set(state, DenseArrayBase.from_list(i64, limit))
        self.strides.set(state, DenseArrayBase.from_list(i64, stride))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        start_indices = cast(
            DenseArrayBase[IntegerType], self.start_indices.get(op)
        ).get_values()
        limit_indices = cast(
            DenseArrayBase[IntegerType], self.limit_indices.get(op)
        ).get_values()
        strides_vals = cast(
            DenseArrayBase[IntegerType], self.strides.get(op)
        ).get_values()
        state.print_whitespace(printer)
        with printer.in_square_brackets():
            # If we're printing invalid IR, this can't be parsed back!
            if len(start_indices) != len(limit_indices) or len(start_indices) != len(
                strides_vals
            ):
                slice_ranges_string = ""
                slice_ranges_string += "start_indices: " + ", ".join(
                    str(x) for x in start_indices
                )
                slice_ranges_string += ", limit_indices: " + ", ".join(
                    str(x) for x in limit_indices
                )
                slice_ranges_string += ", strides: " + ", ".join(
                    str(x) for x in strides_vals
                )
                printer.print_string(slice_ranges_string)
            else:

                def print_range(values: tuple[int, int, int]) -> None:
                    start, limit, stride = values
                    suffix = "" if stride == 1 else f":{stride}"
                    printer.print_string(f"{start}:{limit}{suffix}")

                printer.print_list(
                    list(zip(start_indices, limit_indices, strides_vals)), print_range
                )


@irdl_custom_directive
class DotDimensionNumbers(CustomDirective):
    """
    Custom directive for stablehlo.dot_general dimension numbers.

    Format: `batching_dims = [..] x [..], contracting_dims = [..] x [..]`.
    """

    dimension_numbers: AttributeVariable

    @staticmethod
    def _parse_lhs_rhs_dims(
        parser: Parser,
    ) -> tuple[ArrayAttr[IntegerAttr], ArrayAttr[IntegerAttr]]:
        """Parse `[...] x [...]` into two ArrayAttr[IntegerAttr] attributes."""
        lhs_dims = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: IntegerAttr(parser.parse_integer(), i64)
        )
        parser.parse_keyword("x")
        rhs_dims = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, lambda: IntegerAttr(parser.parse_integer(), i64)
        )
        return ArrayAttr(lhs_dims), ArrayAttr(rhs_dims)

    @staticmethod
    def _print_lhs_rhs_dims(
        printer: Printer,
        lhs_dims: ArrayAttr[IntegerAttr],
        rhs_dims: ArrayAttr[IntegerAttr],
    ) -> None:
        """Print two ArrayAttr[IntegerAttr] as `[...] x [...]`."""
        with printer.in_square_brackets():
            printer.print_list(
                lhs_dims.data, lambda dim: printer.print_int(dim.value.data)
            )
        printer.print_string(" x ")
        with printer.in_square_brackets():
            printer.print_list(
                rhs_dims.data, lambda dim: printer.print_int(dim.value.data)
            )

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        """Parse `batching_dims = [..] x [..], contracting_dims = [..] x [..]`."""
        # Optional `batching_dims = [..] x [..]`.
        lhs_batching, rhs_batching = ArrayAttr(()), ArrayAttr(())
        if parser.parse_optional_keyword("batching_dims") is not None:
            parser.parse_punctuation("=")
            lhs_batching, rhs_batching = self._parse_lhs_rhs_dims(parser)
            parser.parse_punctuation(",")
        # Required `contracting_dims = [..] x [..]`.
        parser.parse_keyword("contracting_dims")
        parser.parse_punctuation("=")
        lhs_contracting, rhs_contracting = self._parse_lhs_rhs_dims(parser)
        self.dimension_numbers.set(
            state,
            DotAttr(
                cast(ArrayAttr[IntegerAttr[I64]], lhs_batching),
                cast(ArrayAttr[IntegerAttr[I64]], rhs_batching),
                cast(ArrayAttr[IntegerAttr[I64]], lhs_contracting),
                cast(ArrayAttr[IntegerAttr[I64]], rhs_contracting),
            ),
        )
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        dims = cast(DotAttr, self.dimension_numbers.get(op))
        state.print_whitespace(printer)
        if dims.lhs_batching_dimensions.data or dims.rhs_batching_dimensions.data:
            printer.print_string("batching_dims = ")
            self._print_lhs_rhs_dims(
                printer, dims.lhs_batching_dimensions, dims.rhs_batching_dimensions
            )
            printer.print_string(", ")
        printer.print_string("contracting_dims = ")
        self._print_lhs_rhs_dims(
            printer, dims.lhs_contracting_dimensions, dims.rhs_contracting_dimensions
        )


@irdl_custom_directive
class PrecisionConfigAndAlgorithm(CustomDirective):
    """
    Custom directive for optional `precision` and `algorithm` clauses of
    stablehlo.dot_general.
    """

    precision_config: AttributeVariable
    algorithm: AttributeVariable

    @staticmethod
    def _parse_precision_config(parser: Parser) -> ArrayAttr[PrecisionAttr]:
        """Parse the precision config as `precision = [...]`."""
        parser.parse_keyword("precision")
        parser.parse_punctuation("=")
        precision_tokens = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_identifier
        )
        precision_attrs: list[PrecisionAttr] = []
        for token in precision_tokens:
            try:
                precision_attrs.append(PrecisionAttr(Precision(token)))
            except ValueError:
                parser.raise_error(f"unknown precision enum '{token}'")
        return ArrayAttr(precision_attrs)

    @staticmethod
    def _print_precision_config(
        printer: Printer, precision_config: ArrayAttr[PrecisionAttr]
    ) -> None:
        """Print precision config as `precision = [..]`."""
        printer.print_string(", precision = [")
        printer.print_list(
            precision_config.data,
            lambda attr: printer.print_string(attr.data.value),
        )
        printer.print_string("]")

    @staticmethod
    def _parse_algorithm_payload(parser: Parser) -> DotAlgorithmAttr:
        """Parse the algorithm payload as `algorithm = {..}`."""
        (
            lhs_precision_type,
            rhs_precision_type,
            accumulation_type,
            lhs_component_count,
            rhs_component_count,
            num_primitive_operations,
            allow_imprecise_accumulation,
        ) = DotAlgorithmAttr.parse_parameters(cast(Any, parser))
        return DotAlgorithmAttr(
            lhs_precision_type,
            rhs_precision_type,
            accumulation_type,
            cast(IntegerAttr[I64], lhs_component_count),
            cast(IntegerAttr[I64], rhs_component_count),
            cast(IntegerAttr[I64], num_primitive_operations),
            cast(BoolAttr, allow_imprecise_accumulation),
        )

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if parser.parse_optional_punctuation(",") is None:
            return True

        # `algorithm = ...` can appear by itself.
        if parser.parse_optional_keyword("algorithm") is not None:
            parser.parse_punctuation("=")
            self.algorithm.set(state, self._parse_algorithm_payload(parser))
            return True

        # Otherwise parse `precision = [...]` first.
        self.precision_config.set(state, self._parse_precision_config(parser))

        # A trailing algorithm clause is optional.
        if parser.parse_optional_punctuation(",") is None:
            return True

        parser.parse_keyword("algorithm")
        parser.parse_punctuation("=")
        self.algorithm.set(state, self._parse_algorithm_payload(parser))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        precision_config = cast(
            ArrayAttr[PrecisionAttr] | None, self.precision_config.get(op)
        )
        algorithm = cast(DotAlgorithmAttr | None, self.algorithm.get(op))
        if precision_config is not None:
            self._print_precision_config(printer, precision_config)
        if algorithm is not None:
            printer.print_string(", algorithm = ")
            algorithm.print_parameters(printer)
