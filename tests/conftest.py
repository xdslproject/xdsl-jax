"""Pytest configuration for xdsl-jax tests."""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUntypedFunctionDecorator=false

from io import StringIO

from filecheck.finput import FInput
from filecheck.matcher import Matcher
from filecheck.options import parse_argv_options
from filecheck.parser import Parser, pattern_for_opts
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.parser import Parser as XDSLParser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.printer import Printer

import pytest


def _run_filecheck_impl(
    program_str: str,
    pipeline: tuple[ModulePass, ...] = (),
    verify: bool = False,
    roundtrip: bool = False,
) -> None:
    """Run filecheck on an xDSL module, comparing it to a program string containing
    filecheck directives."""
    ctx = Context()
    # Register all needed dialects
    ctx.load_dialect(test.Test)

    # Load StableHLO dialect from xdsl-jax
    from xdsl_jax.dialects.stablehlo import StableHLO

    ctx.load_dialect(StableHLO)

    # Load commonly used dialects
    from xdsl.dialects import arith, builtin, func

    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)

    parser = XDSLParser(ctx, program_str)
    xdsl_module = parser.parse_module()

    if roundtrip:
        # Print generic format
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(xdsl_module)
        parser = XDSLParser(ctx, stream.getvalue())
        xdsl_module = parser.parse_module()

    if verify:
        xdsl_module.verify()

    pass_pipeline = PassPipeline(pipeline)
    pass_pipeline.apply(ctx, xdsl_module)

    if verify:
        xdsl_module.verify()

    stream = StringIO()
    Printer(stream).print_op(xdsl_module)
    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", stream.getvalue()),
        Parser(opts, StringIO(program_str), *pattern_for_opts(opts)),
    )

    exit_code = matcher.run()
    assert exit_code == 0, f"""
        filecheck failed with exit code {exit_code}.

        Original program string:
        {program_str}

        Parsed module:
        {stream.getvalue()}
    """


@pytest.fixture
def run_filecheck():
    """Fixture to run filecheck on an xDSL module.

    This fixture uses FileCheck to verify the correctness of a parsed MLIR string. Testers
    can provide a pass pipeline to transform the IR, and verify correctness by including
    FileCheck directives as comments in the input program string.

    Args:
        program_str (str): The MLIR string containing the input program and FileCheck directives
        pipeline (tuple[ModulePass]): A sequence containing all passes that should be applied
            before running FileCheck
        verify (bool): Whether or not to verify the IR after parsing and transforming.
            ``False`` by default.
        roundtrip (bool): Whether or not to use round-trip testing. This is useful for dialect
            tests to verify that xDSL both parses and prints the IR correctly. If ``True``, we parse
            the program string into an xDSL module, print it in generic format, and then parse the
            generic program string back to an xDSL module. ``False`` by default.
    """
    return _run_filecheck_impl
