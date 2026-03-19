// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: JAX_ROUNDTRIP
// RUN: JAX_GENERIC_ROUNDTRIP
// RUN: XDSL_JAX_ROUNDTRIP
// RUN: XDSL_JAX_GENERIC_ROUNDTRIP

// CHECK: %[[OPERAND:.*]] = "test.op"() : () -> tensor<1x3xi64>
// CHECK: %[[OUT_DIMS:.*]] = "test.op"() : () -> tensor<3xi64>
// CHECK: %[[DYNAMIC_BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim %[[OPERAND]], %[[OUT_DIMS]], dims = [2, 1] : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
// CHECK-GENERIC: %[[OPERAND:.*]] = "test.op"() : () -> tensor<1x3xi64>
// CHECK-GENERIC: %[[OUT_DIMS:.*]] = "test.op"() : () -> tensor<3xi64>
// CHECK-GENERIC: %[[DYNAMIC_BCAST:.*]] = "stablehlo.dynamic_broadcast_in_dim"(%[[OPERAND]], %[[OUT_DIMS]]) <{broadcast_dimensions = array<i64: 2, 1>}> : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
%operand = "test.op"() : () -> tensor<1x3xi64>
%out_dims = "test.op"() : () -> tensor<3xi64>
%dynamic_bcast = stablehlo.dynamic_broadcast_in_dim %operand, %out_dims, dims = [2, 1] : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
