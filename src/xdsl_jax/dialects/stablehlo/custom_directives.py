"""
Custom directives for the StableHLO dialect.
"""

from xdsl.irdl import IRDLOperation
from xdsl.irdl.declarative_assembly_format import (
    CustomDirective,
    FunctionalTypeDirective,
    ParsingState,
    PrintingState,
    TypeDirective,
    irdl_custom_directive,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


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

    operand_type: TypeDirective
    result_type: TypeDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        # Try to parse a function type first
        functional_type = FunctionalTypeDirective(
            self.operand_type.inner, self.result_type.inner
        )
        if functional_type.parse(parser, state):
            return True

        # Single type: applies to both operand and result
        single_type = parser.parse_type()
        self.operand_type.set(state, (single_type,))
        self.result_type.set(state, (single_type,))
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operand_types = self.operand_type.get(op)
        result_types = self.result_type.get(op)

        state.print_whitespace(printer)

        # `() -> a` or `(a, a, ...) -> a` prints `a`
        result_type = result_types[0]
        if not operand_types or all(t == result_type for t in operand_types):
            printer.print_attribute(result_type)
            return

        # Fall back to generic
        printer.print_function_type(operand_types, result_types)
