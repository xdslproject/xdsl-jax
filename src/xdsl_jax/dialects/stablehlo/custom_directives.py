"""
Custom directives for the StableHLO dialect.
"""

from typing import cast

from xdsl.dialects.builtin import ComplexType, DenseIntOrFPElementsAttr, TensorType
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
