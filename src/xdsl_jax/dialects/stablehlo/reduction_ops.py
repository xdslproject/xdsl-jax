"""
Reduction operations for the StableHLO dialect.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from xdsl.dialects.builtin import (
    AnyTensorType,
    DenseArrayBase,
    TensorType,
    i64,
)
from xdsl.ir import Attribute, BlockArgument, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.irdl.operations import SameVariadicOperandSize
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.traits import (
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.type import get_element_type_or_self, have_compatible_shape

from .ops import ReturnOp
from .traits import RecursivelySpeculatableIfStaticDimInOutputIsStaticInInput


def _parse_reduce_operand_pairs(
    parser: Parser,
) -> tuple[tuple[UnresolvedOperand, UnresolvedOperand], ...]:
    """Parse the operand pairs of the reduce operation to the parser.
    Specified as:
    `(operand init: init_operand), (operand2 init: init2), ...`
    """

    def parse_operand_pair() -> tuple[UnresolvedOperand, UnresolvedOperand]:
        parser.parse_punctuation("(")
        operand = parser.parse_unresolved_operand()
        parser.parse_keyword("init")
        parser.parse_punctuation(":")
        init_operand = parser.parse_unresolved_operand()
        parser.parse_punctuation(")")
        return operand, init_operand

    return tuple(
        parser.parse_comma_separated_list(Parser.Delimiter.NONE, parse_operand_pair)
    )


def _parse_reduce_dimensions(
    parser: Parser,
) -> list[int]:
    """Parse the dimensions of the reduce operation to the parser.
    Specified as:
    `across dimensions = [dim0, dim1, ...]`
    """
    parser.parse_keyword("across")
    parser.parse_keyword("dimensions")
    parser.parse_punctuation("=")
    dimensions = parser.parse_comma_separated_list(
        Parser.Delimiter.SQUARE, lambda: parser.parse_integer()
    )
    return dimensions


def _parse_reduce_reducer_args(parser: Parser) -> tuple[Parser.Argument, ...]:
    """Parse the reducer arguments of the reduce operation to the parser.
    Specified as:
    `(arg : type, arg : type) for each pair`
    """
    reducer_args: list[Parser.Argument] = []
    while parser.parse_optional_punctuation("(") is not None:
        arg0 = parser.parse_argument(expect_type=True)
        parser.parse_punctuation(",")
        arg1 = parser.parse_argument(expect_type=True)
        parser.parse_punctuation(")")
        reducer_args.extend((arg0, arg1))
    return tuple(reducer_args)


def _print_reduce_operand_pairs(
    printer: Printer,
    inputs: Sequence[SSAValue],
    init_values: Sequence[SSAValue],
) -> None:
    """Print the operand pairs of the reduce operation to the printer.
    Specified as:
    `(operand init: init_operand), (operand2 init: init2), ...`
    """

    def print_operand_pair(pair: tuple[SSAValue, SSAValue]) -> None:
        with printer.in_parens():
            printer.print_ssa_value(pair[0])
            printer.print_string(" init: ")
            printer.print_ssa_value(pair[1])

    printer.print_list(zip(inputs, init_values), print_operand_pair)


def _print_reduce_dimensions_and_attrs(
    printer: Printer,
    dimensions: DenseArrayBase | None,
    attributes: Mapping[str, Attribute],
) -> None:
    """Print the dimensions and attributes of the reduce operation to the printer.
    Specified as:
    `across dimensions = [dim0, dim1, ...]`
    """
    printer.print_string(" across dimensions = [")
    dims = cast(tuple[int, ...], cast(Any, dimensions).get_values())
    printer.print_list(dims, lambda d: printer.print_string(str(d)))
    printer.print_string("]")
    if attributes:
        printer.print_string(" ")
        printer.print_op_attributes(attributes)


def _print_reduce_function_type(
    printer: Printer,
    inputs: Sequence[SSAValue],
    init_values: Sequence[SSAValue],
    results: Sequence[SSAValue],
) -> None:
    """Print the function type of the reduce operation to the printer."""
    printer.print_string(" : ")
    operand_types = [v.type for v in inputs] + [v.type for v in init_values]
    result_types = [r.type for r in results]
    printer.print_function_type(operand_types, result_types)


def _print_reduce_reducer(
    printer: Printer,
    body: Region,
    num_pairs: int,
) -> None:
    """Print the reducer region of the reduce operation to the printer.
    Specified as:
    `reducer (arg : type, arg : type) for each pair`
    """
    printer.print_string("\nreducer ")
    block = body.blocks[0]

    def print_reducer_pair(pair: tuple[BlockArgument, BlockArgument]) -> None:
        with printer.in_parens():
            printer.print_block_argument(pair[0])
            printer.print_string(", ")
            printer.print_block_argument(pair[1])

    reducer_pairs = zip(block.args[:num_pairs], block.args[num_pairs:])
    printer.print_list(reducer_pairs, print_reducer_pair, delimiter=" ")
    printer.print_string(" ")
    printer.print_region(body, print_entry_block_args=False)


def _verify_reduce_counts(
    inputs: Sequence[SSAValue],
    init_values: Sequence[SSAValue],
    result: Sequence[SSAValue],
) -> int:
    """Verify the number of input/init_value pairs to be the same according
    to the StableHLO specification. Specified as:
    (C3) `size(inputs) = size(init_values) = size(results) = N`
    """
    num_pairs = len(inputs)
    if num_pairs == 0:
        raise VerifyException("Reduce op expects at least one input/init_value pair")

    if len(init_values) != num_pairs or len(result) != num_pairs:
        raise VerifyException(
            "Reduce op requires the same number of inputs, init_values, and results"
        )

    return num_pairs


def _verify_reduce_input_shapes(
    input_types: Sequence[TensorType[Attribute]],
) -> tuple[tuple[int, ...], int]:
    """Verify all inputs to have the compatible shape according
    to the StableHLO specification. Specified as:
    (C1) `same(shape(inputs))`
    """
    first_input_type = input_types[0]
    for in_type in input_types[1:]:
        if not have_compatible_shape(first_input_type, in_type):
            raise VerifyException("Reduce inputs must have the same shape.")

    input_shape = first_input_type.get_shape()
    return input_shape, len(input_shape)


def _verify_reduce_dimensions(dims: Sequence[int], rank: int) -> None:
    """Verify the dimensions of the reduce operation to have the correct dimensions
    according to the StableHLO specification. Specified as:
    (C4) `0 <= dimensions < rank(inputs[0])`
    (C5) `is_unique(dimensions)`
    """
    if len(set(dims)) != len(dims):
        raise VerifyException(f"Reduce dimensions must be unique, got {dims}")
    for dim in dims:
        if dim < 0 or dim >= rank:
            raise VerifyException(
                f"Reduce dimension {dim} out of range for rank {rank}"
            )


def _verify_reduce_init_values(
    init_types: Sequence[TensorType[Attribute]],
    input_elem_types: Sequence[Attribute],
) -> None:
    """Verify the init values of the reduce operation to have the correct element
    type according to the StableHLO specification. Specified as:
    (C2) `element_type(inputs) = element_type(init_values)`
    """
    for idx, (init_type, input_elem_type) in enumerate(
        zip(init_types, input_elem_types)
    ):
        if init_type.get_shape():
            raise VerifyException(
                "Reduce init_values must be 0-dimensional tensors; "
                f"found rank {len(init_type.get_shape())} at index {idx}"
            )
        if get_element_type_or_self(init_type) != input_elem_type:
            raise VerifyException(
                f"Reduce input and init_value element types must match at index {idx}"
            )


def _verify_reduce_results(
    result_types: Sequence[TensorType[Attribute]],
    input_shape: Sequence[int],
    dims: Sequence[int],
    input_elem_types: Sequence[Attribute],
) -> None:
    """Verify the results of the reduce operation to have the correct shape and
    element type according to the StableHLO specification. Specified as:
    (C7) `shape(results) = shape(inputs)` except that the dimension
    sizes of inputs corresponding to dimensions are not included.
    (C8) `element_type(results[i]) = Ei` for all `i` in `[0,N)`.
    where `Ei` is the element type of the input tensors.
    """
    dims_set = set(dims)
    expected_shape = tuple(
        dim for i, dim in enumerate(input_shape) if i not in dims_set
    )
    for idx, res_type in enumerate(result_types):
        if res_type.get_shape() != expected_shape:
            raise VerifyException(
                "Reduce result shape mismatch at index "
                f"{idx}: expected {expected_shape}, got {res_type.get_shape()}"
            )
        if get_element_type_or_self(res_type) != input_elem_types[idx]:
            raise VerifyException(
                "Reduce result element types must match input element types "
                f"at index {idx}"
            )


def _verify_reduce_body(
    body: Region,
    num_pairs: int,
    input_elem_types: Sequence[Attribute],
) -> None:
    """Verify the reducer region of the reduce operation to have the correct
    body type according to the StableHLO specification. Specified as:
    `(tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ...,tensor<EN-1>)`
    `-> (tensor<E0>, ..., tensor<EN-1>)`
    where `Ei` is the element type of the input tensors.
    """
    block = body.blocks[0]
    if len(block.args) != 2 * num_pairs:
        raise VerifyException(
            f"Reduce body must take {2 * num_pairs} arguments, got {len(block.args)}"
        )

    for idx in range(num_pairs):
        arg0_type = block.args[idx].type
        arg1_type = block.args[idx + num_pairs].type
        if not isinstance(arg0_type, TensorType) or not isinstance(
            arg1_type, TensorType
        ):
            raise VerifyException(
                f"Reduce body arguments for pair {idx} must be tensor types"
            )
        arg0_type = cast(TensorType[Attribute], arg0_type)
        arg1_type = cast(TensorType[Attribute], arg1_type)
        if arg0_type.get_shape() or arg1_type.get_shape():
            raise VerifyException(
                f"Reduce body arguments for pair {idx} must be 0-dim tensors"
            )
        elem_type = input_elem_types[idx]
        if (
            get_element_type_or_self(arg0_type) != elem_type
            or get_element_type_or_self(arg1_type) != elem_type
        ):
            raise VerifyException(
                "Reduce body argument element types must match input element types "
                f"at index {idx}"
            )

    terminator = cast(ReturnOp, block.last_op)
    if len(terminator.input) != num_pairs:
        raise VerifyException(
            f"Reduce body must return {num_pairs} values, got {len(terminator.input)}"
        )
    for idx, ret_val in enumerate(terminator.input):
        if not isinstance(ret_val.type, TensorType):
            raise VerifyException(
                f"Reduce body return value at index {idx} must be a tensor"
            )
        ret_val_typed = cast(SSAValue[TensorType[Attribute]], ret_val)
        ret_type = ret_val_typed.type
        if ret_type.get_shape():
            raise VerifyException(
                f"Reduce body return value at index {idx} must be 0-dim tensor"
            )
        if get_element_type_or_self(ret_type) != input_elem_types[idx]:
            raise VerifyException(
                "Reduce body return element types must match input element types "
                f"at index {idx}"
            )


@irdl_op_definition
class ReduceOp(IRDLOperation):
    """
    Applies a reduction function ``body`` to ``inputs`` and ``init_values`` along the
    ``dimensions`` and produces a ``result`` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    ```
    """

    name = "stablehlo.reduce"

    inputs = var_operand_def(AnyTensorType)
    init_values = var_operand_def(AnyTensorType)
    dimensions = prop_def(DenseArrayBase.constr(i64))
    result = var_result_def(AnyTensorType)
    body = region_def("single_block")

    irdl_options = (SameVariadicOperandSize(),)

    traits = traits_def(
        RecursiveMemoryEffect(),
        RecursivelySpeculatableIfStaticDimInOutputIsStaticInInput(),
        SingleBlockImplicitTerminator(ReturnOp),
    )

    def print(self, printer: Printer) -> None:
        """Custom print method for ReduceOp."""
        num_pairs = len(self.inputs)
        _print_reduce_operand_pairs(printer, self.inputs, self.init_values)
        _print_reduce_dimensions_and_attrs(
            printer,
            self.dimensions,
            self.attributes,
        )
        _print_reduce_function_type(
            printer,
            self.inputs,
            self.init_values,
            self.result,
        )
        _print_reduce_reducer(printer, self.body, num_pairs)

    @classmethod
    def parse(cls, parser: Parser) -> "ReduceOp":
        """Custom parse method for ReduceOp."""

        # Parse (operand init: init_operand), (operand2 init: init2), ...
        operand_pairs = _parse_reduce_operand_pairs(parser)
        operands, init_operands = zip(*operand_pairs)
        flattened_operands = operands + init_operands
        num_pairs = len(operand_pairs)
        dimensions = _parse_reduce_dimensions(parser)
        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()
        # reducer (arg : type, arg : type) for each pair
        parser.parse_keyword("reducer")
        arguments = _parse_reduce_reducer_args(parser)
        body = parser.parse_region(arguments)
        # Resolve operands: inputs then init_values
        resolved = parser.resolve_operands(
            flattened_operands,
            func_type.inputs.data,
            parser.pos,
        )

        return cls.build(
            operands=[resolved[:num_pairs], resolved[num_pairs:]],
            result_types=[func_type.outputs.data],
            regions=[body],
            properties={"dimensions": DenseArrayBase.from_list(i64, dimensions)},
        )

    def verify_(self) -> None:
        """Custom verify method for ReduceOp."""
        input_types = [cast(TensorType[Attribute], v.type) for v in self.inputs]
        init_types = [cast(TensorType[Attribute], v.type) for v in self.init_values]
        result_types = [r.type for r in self.result]
        input_elem_types = [get_element_type_or_self(t) for t in input_types]
        dims = self.dimensions.get_values()
        num_pairs = _verify_reduce_counts(self.inputs, self.init_values, self.result)
        input_shape, rank = _verify_reduce_input_shapes(input_types)
        _verify_reduce_dimensions(dims, rank)
        _verify_reduce_init_values(init_types, input_elem_types)
        _verify_reduce_results(result_types, input_shape, dims, input_elem_types)
        _verify_reduce_body(self.body, num_pairs, input_elem_types)
