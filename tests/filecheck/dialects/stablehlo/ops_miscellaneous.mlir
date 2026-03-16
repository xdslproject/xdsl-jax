// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: %[[CONSTANT:.*]] = stablehlo.constant dense<{{.*}}> : tensor<2x2xf32>
// CHECK-GENERIC: %[[CONSTANT:.*]] = "stablehlo.constant"() <{value = dense<{{.*}}> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
%constant = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>

// CHECK: %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<4x5xi32>
// CHECK-GENERIC: %[[IOTA:.*]] = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4x5xi32>
%iota = stablehlo.iota dim = 0 : tensor<4x5xi32>
