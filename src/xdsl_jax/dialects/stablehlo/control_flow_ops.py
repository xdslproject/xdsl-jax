"""
Control flow operations for the StableHLO dialect.
"""

from __future__ import annotations

from typing import cast

from xdsl.dialects.builtin import TensorType, i1
from xdsl.dialects.utils.format import parse_assignment, print_assignment
from xdsl.ir import Attribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    Pure,
    RecursivelySpeculatable,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException

from .custom_directives import PairwiseOpType
from .ops import ReturnOp
from .traits import have_compatible_type_sequences
from .types import PredTensorType, TensorOrTokenType


def _is_zero_rank_i1_tensor(attr: Attribute) -> bool:
    """Verify that the attribute is a zero-ranked tensor of i1."""
    if not isinstance(attr, TensorType):
        return False
    tensor_attr = cast(TensorType[Attribute], attr)
    return tensor_attr.get_num_dims() == 0 and tensor_attr.element_type == i1


@irdl_op_definition
class IfOp(IRDLOperation):
    """
    Produces the output from executing exactly one branch from `true_branch` or
    `false_branch` depending on the value of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if

    Example:
    %result = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%result_true_branch) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
    """

    name = "stablehlo.if"

    pred = operand_def(PredTensorType)

    res = var_result_def(TensorOrTokenType)

    true_branch = region_def("single_block")

    false_branch = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
        SingleBlockImplicitTerminator(ReturnOp),
    )


@irdl_op_definition
class WhileOp(IRDLOperation):
    """
    Produces the output from executing `body` function 0 or more times while the
    `cond` function outputs `true`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while

    Example:
    ```mlir
    %results0, %results1 = stablehlo.while(%arg0 = %init_i, %arg1 = %init_sum) :
      tensor<i64>, tensor<i64>
    cond {
      %cond = stablehlo.compare LT, %arg0, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %new_sum = stablehlo.add %arg1, %one : tensor<i64>
      %new_i = stablehlo.add %arg0, %one : tensor<i64>
      stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }
    """

    name = "stablehlo.while"

    operand = var_operand_def(TensorOrTokenType)

    res = var_result_def(TensorOrTokenType)

    cond = region_def("single_block")

    body = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        RecursivelySpeculatable(),
        SingleBlockImplicitTerminator(ReturnOp),
    )

    def _verify_cond_block_args(self, operand_types: tuple[Attribute, ...]) -> None:
        """Verify that operand types are compatible with condition block arguments."""
        cond_arg_types = tuple(arg.type for arg in self.cond.block.args)
        if not have_compatible_type_sequences(operand_types, cond_arg_types):
            raise VerifyException(
                "expect operands to be compatible with condition block arguments "
                f"but got {operand_types} vs {cond_arg_types}"
            )

    def _verify_body_block_args(self, operand_types: tuple[Attribute, ...]) -> None:
        """Verify that operand types are compatible with body block arguments."""
        body_arg_types = tuple(arg.type for arg in self.body.block.args)
        if not have_compatible_type_sequences(operand_types, body_arg_types):
            raise VerifyException(
                "expect operands to be compatible with body block arguments "
                f"but got {operand_types} vs {body_arg_types}"
            )

    def _verify_body_block_return(self, operand_types: tuple[Attribute, ...]) -> None:
        """Verify that the body block has a return op and
        the return types are compatible with the operand types."""
        body_block = self.body.block
        if not isinstance(body_block.last_op, ReturnOp):
            raise VerifyException("The while body-region expected to have a terminator")

        body_return_types = tuple(operand.type for operand in body_block.last_op.input)
        if not have_compatible_type_sequences(operand_types, body_return_types):
            raise VerifyException(
                "expect operands to be compatible with body block return types "
                f"but got {operand_types} vs {body_return_types}"
            )

    def _verify_cond_block_return(self) -> None:
        """Verify that the condition block has a single return op and
        the return type is a zero-ranked tensor of i1."""
        cond_block = self.cond.block
        if not isinstance(cond_block.last_op, ReturnOp):
            raise VerifyException(
                "The while condition-region expected to have a "
                "`stablehlo.return` terminator"
            )

        cond_return_types = tuple(operand.type for operand in cond_block.last_op.input)
        if len(cond_return_types) != 1:
            raise VerifyException(
                "expect condition body returns a single value "
                f"but got {len(cond_return_types)}"
            )

        cond_return_type = cond_return_types[0]
        if not _is_zero_rank_i1_tensor(cond_return_type):
            raise VerifyException(
                "expect condition block return a zero-ranked tensor of i1 but got "
                f"{cond_return_type}"
            )

    def verify_(self) -> None:
        operand_types = tuple(self.operand_types)
        self._verify_cond_block_args(operand_types)
        self._verify_body_block_args(operand_types)
        self._verify_body_block_return(operand_types)
        self._verify_cond_block_return()

    def print(self, printer: Printer) -> None:
        body_args = self.body.block.args if self.body.block else ()
        # Print block arguments and operands
        with printer.in_parens():
            printer.print_list(
                zip(body_args, self.operand),
                lambda pair: print_assignment(printer, pair[0], pair[1]),
            )
        # Print types
        if self.operand:
            printer.print_string(" : ")
            printer.print_list(
                (operand.type for operand in self.operand), printer.print_attribute
            )
        # Print attributes
        if self.attributes:
            printer.print_op_attributes(self.attributes, print_keyword=True)
        # Print regions
        printer.print_string("\ncond ")
        printer.print_region(self.cond, print_entry_block_args=False)
        printer.print_string(" do ")
        printer.print_region(self.body, print_entry_block_args=False)

    @classmethod
    def parse(cls, parser: Parser) -> WhileOp:
        # Parse arguments and operands
        pairs = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parse_assignment(parser)
        )
        # Unpack pairs into iter_args and operands
        if pairs:
            iter_args, operands = zip(*pairs)
        else:
            iter_args = ()
            operands = ()
        # Parse types
        types: tuple[Attribute, ...] = ()
        if operands:
            parser.parse_punctuation(":")
            types = tuple(
                parser.parse_comma_separated_list(
                    parser.Delimiter.NONE, parser.parse_type
                )
            )
        # Resolve operands
        resolved_operands = parser.resolve_operands(operands, types, parser.pos)
        arguments = tuple(arg.resolve(t) for arg, t in zip(iter_args, types))
        # Parse attributes
        dict_attr = parser.parse_optional_attr_dict_with_keyword()
        attrs: dict[str, Attribute] = dict(dict_attr.data) if dict_attr else {}
        # Parse regions
        parser.parse_keyword("cond")
        cond = parser.parse_region(arguments=arguments)
        parser.parse_keyword("do")
        body = parser.parse_region(arguments=arguments)
        # Build operation
        return cls.build(
            operands=[resolved_operands],
            result_types=[types],
            regions=[cond, body],
            attributes=attrs,
        )


@irdl_op_definition
class OptimizationBarrierOp(IRDLOperation):
    """
    Ensures that the operations that produce the `operand` are executed before any
    operations that depend on the `result` and prevents compiler transformations
    from moving operations across the barrier. Other than that, the operation is
    an identity, i.e. `result` = `operand`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#optimization_barrier

    Example:
    ```mlir
    %result0, %result1 = stablehlo.optimization_barrier %operand0, %operand1
     : tensor<f32>, tensor<f32>
    ```
    """

    name = "stablehlo.optimization_barrier"

    operand = var_operand_def(TensorOrTokenType)

    res = var_result_def(TensorOrTokenType)

    traits = traits_def(
        Pure(),
    )

    def verify_(self) -> None:
        """Same number of operands and results, and pairwise same types."""
        num_operands = len(self.operand)
        num_results = len(self.res)
        if num_operands != num_results:
            raise VerifyException(
                f"requires the same number of operands and results "
                f"(got {num_operands} operands and {num_results} results)"
            )
        for idx in range(num_operands):
            if self.operand[idx].type != self.res[idx].type:
                raise VerifyException(
                    f"requires the same type for operand and result at index {idx} "
                    f"(got {self.operand[idx].type} vs {self.res[idx].type})"
                )

    assembly_format = (
        "$operand attr-dict `:` custom<PairwiseOpType>(type($operand), type($res))"
    )
    custom_directives = (PairwiseOpType,)
