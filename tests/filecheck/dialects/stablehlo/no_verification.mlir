// RUN: xdsl-opt %s --disable-verify | filecheck %s

%slice_input = "test.op"() : () -> tensor<3x8xi64>

// CHECK: %slice_mismatch = stablehlo.slice %slice_input [start_indices: 1, 4, limit_indices: 3, strides: 1, 2] : (tensor<3x8xi64>) -> tensor<2x2xi64>
%slice_mismatch = "stablehlo.slice"(%slice_input) <{start_indices = array<i64: 1, 4>, limit_indices = array<i64: 3>, strides = array<i64: 1, 2>}> : (tensor<3x8xi64>) -> tensor<2x2xi64>
