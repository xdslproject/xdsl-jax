"""Test the constraints defined within the xdsl_extras module."""

from typing import TypeVar

from xdsl.dialects.builtin import MemRefType, TensorType, TupleType, i1, i32
from xdsl.irdl import BaseAttr, TypeVarConstraint
from xdsl.irdl.constraints import AnyOf, ConstraintContext
from xdsl.utils.exceptions import VerifyException

import pytest
from xdsl_jax.dialects.stablehlo import TokenType
from xdsl_jax.xdsl_extras import NestedTupleOfConstraint


class TestNestedTupleOfConstraint:
    """Tests for the NestedTupleOfConstraint class."""

    multiple_constraint = NestedTupleOfConstraint(AnyOf([TensorType, TokenType]))
    single_constraint = NestedTupleOfConstraint(TensorType)

    def test_nested_tuple_of_constraint(self):
        """Test that the properties of NestedTupleOfConstraint object are correct."""
        assert isinstance(self.multiple_constraint.elem_constraint, AnyOf)
        assert hasattr(self.multiple_constraint.elem_constraint, "attr_constrs")

        assert not isinstance(self.single_constraint.elem_constraint, AnyOf)
        assert self.single_constraint.elem_constraint == BaseAttr(TensorType)

    def test_single_constraint_accepts_nested(self):
        """Test that nested tuples with only TensorType are accepted."""
        tensor1 = TensorType(i32, [2])
        tensor2 = TensorType(i1, [1])
        inner = TupleType((tensor1,))
        outer = TupleType((tensor2, inner))
        self.single_constraint.verify(outer, ConstraintContext())

    def test_single_constraint_rejects_disallowed_type(self):
        """Test that a tuple with non-TensorType elements raises a VerifyException."""
        tensor = TensorType(i32, [2])
        token = TokenType()
        invalid_tup = TupleType((tensor, token))
        with pytest.raises(
            VerifyException,
            match="tuple leaf 1 failed constraint:",
        ):
            self.single_constraint.verify(invalid_tup, ConstraintContext())

    def test_nested_tuple_of_constraint_accepts_nested(self):
        """Test that nested tuples are accepted."""
        tensor1 = TensorType(i32, [2])
        tensor2 = TensorType(i1, [1])
        token = TokenType()
        inner = TupleType((token, tensor2))
        outer = TupleType((tensor1, inner))
        self.multiple_constraint.verify(outer, ConstraintContext())

    def test_nested_tuple_of_constraint_rejects_disallowed_type(self):
        """Test that a tuple with a disallowed type raises a VerifyException."""
        tensor = TensorType(i32, [2])
        memref = MemRefType(i32, [2])
        tup = TupleType((tensor, memref))
        with pytest.raises(
            VerifyException,
            match="tuple leaf 1 failed constraint: memref<2xi32>",
        ):
            self.multiple_constraint.verify(tup, ConstraintContext())

    def test_nested_tuple_of_constraint_mapping_type_vars(self):
        """Test that mapping_type_vars correctly replaces type variables in nested
        constraints."""
        _T = TypeVar("_T")

        constraint = NestedTupleOfConstraint(
            AnyOf([TypeVarConstraint(_T, BaseAttr(TensorType)), BaseAttr(TokenType)])
        )

        with pytest.raises(KeyError, match="Mapping value missing for type var"):
            constraint.mapping_type_vars({})

        expected = NestedTupleOfConstraint(
            AnyOf([BaseAttr(MemRefType), BaseAttr(TokenType)])
        )
        assert constraint.mapping_type_vars({_T: BaseAttr(MemRefType)}) == expected
