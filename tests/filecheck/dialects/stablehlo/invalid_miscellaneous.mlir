// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

// CHECK: Operation does not verify: Iota output must have a static shape.
%iota = stablehlo.iota dim = 0 : tensor<?x3xi32>

// -----

// CHECK: Operation does not verify: Iota does not support scalars.
%iota = stablehlo.iota dim = 0 : tensor<i32>

// -----

// CHECK: Operation does not verify: Iota dimension cannot go beyond the output rank.
%iota = stablehlo.iota dim = 3 : tensor<2x3xi32>
