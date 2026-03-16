// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%dbc2_operand = "test.op"() : () -> tensor<1x3xi64>
%dbc2_out_dims = "test.op"() : () -> tensor<3xi64>
// CHECK: Operation does not verify: broadcast_dimensions size (3) does not match operand rank (2)
%dbc2_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc2_operand, %dbc2_out_dims) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>

// -----

%dbc3_operand = "test.op"() : () -> tensor<2x3x4xi64>
%dbc3_out_dims = "test.op"() : () -> tensor<2xi64>
// CHECK: Operation does not verify: result rank (2) is less than operand rank (3)
%dbc3_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc3_operand, %dbc3_out_dims) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<2x3x4xi64>, tensor<2xi64>) -> tensor<2x3xi64>

// -----

%dbc7_operand = "test.op"() : () -> tensor<1x3xi64>
%dbc7_out_dims = "test.op"() : () -> tensor<2xi64>
// CHECK: Operation does not verify: length of output_dimensions (2) is not compatible with result rank (3)
%dbc7_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc7_operand, %dbc7_out_dims) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<1x3xi64>, tensor<2xi64>) -> tensor<2x3x2xi64>

// -----

%dbc4_operand = "test.op"() : () -> tensor<1x3xi64>
%dbc4_out_dims = "test.op"() : () -> tensor<3xi64>
// CHECK: Operation does not verify: broadcast_dimensions should not have duplicates
%dbc4_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc4_operand, %dbc4_out_dims) {broadcast_dimensions = array<i64: 1, 1>} : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>

// -----

%dbc5a_operand = "test.op"() : () -> tensor<1x3xi64>
%dbc5a_out_dims = "test.op"() : () -> tensor<3xi64>
// CHECK: Operation does not verify: broadcast_dimensions contains invalid value 5 for result with rank 3
%dbc5a_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc5a_operand, %dbc5a_out_dims) {broadcast_dimensions = array<i64: 5, 1>} : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>

// -----

%dbc5b_operand = "test.op"() : () -> tensor<3x4xi64>
%dbc5b_out_dims = "test.op"() : () -> tensor<3xi64>
// CHECK: Operation does not verify: size of operand dimension 0 (3) is not compatible with size of result dimension 0 (2)
%dbc5b_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc5b_operand, %dbc5b_out_dims) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<3x4xi64>, tensor<3xi64>) -> tensor<2x4x5xi64>

// -----

%dbc8_operand = "test.op"() : () -> tensor<1x3xi64>
%dbc8_out_dims = "test.op"() : () -> tensor<3xi64>
// CHECK: Operation does not verify: duplicate expansion hint for at least one operand dimension
%dbc8_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc8_operand, %dbc8_out_dims) {broadcast_dimensions = array<i64: 2, 1>, known_expanding_dimensions = array<i64: 0>, known_nonexpanding_dimensions = array<i64: 0>} : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>

// -----

%dbc9_operand = "test.op"() : () -> tensor<1x3xi64>
%dbc9_out_dims = "test.op"() : () -> tensor<3xi64>
// CHECK: Operation does not verify: hint for expanding dimension 5 does not refer to a valid operand dimension
%dbc9_result = "stablehlo.dynamic_broadcast_in_dim"(%dbc9_operand, %dbc9_out_dims) {broadcast_dimensions = array<i64: 2, 1>, known_expanding_dimensions = array<i64: 5>} : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
