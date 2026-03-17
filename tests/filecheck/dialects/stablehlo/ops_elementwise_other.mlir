// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: JAX_ROUNDTRIP
// RUN: JAX_GENERIC_ROUNDTRIP
// RUN: XDSL_JAX_ROUNDTRIP
// RUN: XDSL_JAX_GENERIC_ROUNDTRIP

// CHECK: %[[PRED:.*]] = "test.op"() : () -> tensor<i1>
// CHECK-GENERIC: %[[PRED:.*]] = "test.op"() : () -> tensor<i1>
%pred = "test.op"() : () -> tensor<i1>
// CHECK: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
%t0 = "test.op"() : () -> tensor<i32>
// CHECK: %[[T5F32:.*]] = "test.op"() : () -> tensor<5xf32>
// CHECK-GENERIC: %[[T5F32:.*]] = "test.op"() : () -> tensor<5xf32>
%t5f32 = "test.op"() : () -> tensor<5xf32>
// CHECK: %[[TDF32:.*]] = "test.op"() : () -> tensor<?xf32>
// CHECK-GENERIC: %[[TDF32:.*]] = "test.op"() : () -> tensor<?xf32>
%tdf32 = "test.op"() : () -> tensor<?xf32>

// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[T0]], %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[CLAMP:.*]] = "stablehlo.clamp"(%[[T0]], %[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
%clamp = stablehlo.clamp %t0, %t0, %t0 : tensor<i32>

// CHECK: %[[BITCAST:.*]] = stablehlo.bitcast_convert %[[T0]] : (tensor<i32>) -> tensor<2xi16>
// CHECK-GENERIC: %[[BITCAST:.*]] = "stablehlo.bitcast_convert"(%[[T0]]) : (tensor<i32>) -> tensor<2xi16>
%bitcast = stablehlo.bitcast_convert %t0 : (tensor<i32>) -> tensor<2xi16>

// CHECK: %[[COMPARE:.*]] = stablehlo.compare EQ, %[[T0]], %[[T0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-GENERIC: %[[COMPARE:.*]] = "stablehlo.compare"(%[[T0]], %[[T0]]) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
%compare = stablehlo.compare EQ, %t0, %t0 : (tensor<i32>, tensor<i32>) -> tensor<i1>

// CHECK: %[[MAP:.*]] = "stablehlo.map"(%[[T5F32]], %[[T5F32]]) <{dimensions = array<i64: 0>}> ({
// CHECK-NEXT: ^bb0(%[[MAP_ARG0:[^ )]+]] : tensor<f32>, %[[MAP_ARG1:[^ )]+]] : tensor<f32>):
// CHECK-NEXT:   %[[MAP_MUL:.*]] = stablehlo.multiply %[[MAP_ARG0]], %[[MAP_ARG1]] : tensor<f32>
// CHECK-NEXT:   stablehlo.return %[[MAP_MUL]] : tensor<f32>
// CHECK-NEXT: }) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
%map = "stablehlo.map"(%t5f32, %t5f32) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %result = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    stablehlo.return %result : tensor<f32>
}) {
  dimensions = array<i64: 0>
} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>

// CHECK: %[[TF64:.*]] = "test.op"() : () -> tensor<f64>
// CHECK-GENERIC: %[[TF64:.*]] = "test.op"() : () -> tensor<f64>
%tf64 = "test.op"() : () -> tensor<f64>
// CHECK: %[[REDUCE_PRECISION:.*]] = stablehlo.reduce_precision %[[TF64]], format = e5m10 : tensor<f64>
// CHECK-GENERIC: %[[REDUCE_PRECISION:.*]] = "stablehlo.reduce_precision"(%[[TF64]]) <{exponent_bits = 5 : i32, mantissa_bits = 10 : i32}> : (tensor<f64>) -> tensor<f64>
%reduce_precision = stablehlo.reduce_precision %tf64, format = e5m10 : tensor<f64>

// CHECK: %[[SELECT:.*]] = stablehlo.select %[[PRED]], %[[T0]], %[[T0]] : tensor<i1>, tensor<i32>
// CHECK-GENERIC: %[[SELECT:.*]] = "stablehlo.select"(%[[PRED]], %[[T0]], %[[T0]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
%select = stablehlo.select %pred, %t0, %t0 : tensor<i1>, tensor<i32>

// CHECK: %[[SELECT_MISMATCH:.*]] = stablehlo.select %[[PRED]], %[[T0]], %[[T0]] : tensor<i1>, tensor<i32>
// CHECK-GENERIC: %[[SELECT_MISMATCH:.*]] = "stablehlo.select"(%[[PRED]], %[[T0]], %[[T0]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
%select_mismatch = stablehlo.select %pred, %t0, %t0 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK: %[[SELECT_FUNCTION_TYPE:.*]] = stablehlo.select %pred, %[[TDF32]], %[[T5F32]] : (tensor<i1>, tensor<?xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK-GENERIC: %[[SELECT_FUNCTION_TYPE:.*]] = "stablehlo.select"(%[[PRED]], %[[TDF32]], %[[T5F32]]) : (tensor<i1>, tensor<?xf32>, tensor<5xf32>) -> tensor<5xf32>
%select_function_type = stablehlo.select %pred, %tdf32, %t5f32 : (tensor<i1>, tensor<?xf32>, tensor<5xf32>) -> tensor<5xf32>
