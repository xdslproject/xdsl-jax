// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%map_input = "test.op"() : () -> tensor<5xf32>
// CHECK: Operation does not verify: expects number of operands to match the arity of map computation, but got: 1 and 2
%map_arity = "stablehlo.map"(%map_input) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    stablehlo.return %arg0 : tensor<f32>
}) {dimensions = array<i64: 0>} : (tensor<5xf32>) -> tensor<5xf32>

// -----

%map_input = "test.op"() : () -> tensor<5xf32>
// CHECK: Operation does not verify: computation arguments must be 0-rank tensor, but got: arg #0 of type tensor<1xf32>
%map_arg_rank = "stablehlo.map"(%map_input) ({
  ^bb0(%arg0: tensor<1xf32>):
    %0 = "test.op"() : () -> tensor<f32>
    stablehlo.return %0 : tensor<f32>
}) {dimensions = array<i64: 0>} : (tensor<5xf32>) -> tensor<5xf32>

// -----

%map_input = "test.op"() : () -> tensor<5xf32>
// CHECK: Operation does not verify: computation must return single output, but got: 2
%map_return_count = "stablehlo.map"(%map_input) ({
  ^bb0(%arg0: tensor<f32>):
    stablehlo.return %arg0, %arg0 : tensor<f32>, tensor<f32>
}) {dimensions = array<i64: 0>} : (tensor<5xf32>) -> tensor<5xf32>

// -----

%map_input = "test.op"() : () -> tensor<5xf32>
// CHECK: Operation does not verify: requires monotonically increasing dimension numbers, but got: (1,)
%map_dims_order = "stablehlo.map"(%map_input) ({
  ^bb0(%arg0: tensor<f32>):
    stablehlo.return %arg0 : tensor<f32>
}) {dimensions = array<i64: 1>} : (tensor<5xf32>) -> tensor<5xf32>

// -----

%pred = "test.op"() : () -> tensor<i1>
%on_true = "test.op"() : () -> tensor<i32>
%on_false = "test.op"() : () -> tensor<i32>
// CHECK: expected functional type or list of two types
%bad_select_type = stablehlo.select %pred, %on_true, %on_false : tensor<i1>, tensor<i32>, tensor<i32>

// -----

%arg_reduce_precision = "test.op"() : () -> tensor<2xf32>
// CHECK: expected exponent mantissa in format e#m#, saw nope
%bad_reduce_precision = stablehlo.reduce_precision %arg_reduce_precision, format = nope : tensor<2xf32>
