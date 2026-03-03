#!/usr/bin/env python3
"""
JAX StableHLO roundtrip test script.

Reads MLIR text containing StableHLO operations, parses it using JAX's MLIR API,
and prints it back out to validate compatibility with JAX's StableHLO implementation.
"""

import sys
from pathlib import Path

from jax._src.interpreters import mlir as jax_mlir


def ir_roundtrip(mlir_text: str) -> str:
    with jax_mlir.make_ir_context() as ctx:  # pyright: ignore
        ctx.allow_unregistered_dialects = True
        module = jax_mlir.ir.Module.parse(mlir_text)  # pyright: ignore
        output = module.operation.get_asm(large_elements_limit=20)  # pyright: ignore
        return output  # pyright: ignore


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="JAX StableHLO roundtrip test script. "
        "Reads MLIR text (from a file or stdin), parses it using JAX's MLIR API, "
        "and prints it back out to validate."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="MLIR file to read. If omitted, reads from stdin.",
    )
    args = parser.parse_args()

    mlir_text = (
        sys.stdin.read() if args.input_file is None else args.input_file.read_text()
    )

    print(ir_roundtrip(mlir_text))


if __name__ == "__main__":
    main()
