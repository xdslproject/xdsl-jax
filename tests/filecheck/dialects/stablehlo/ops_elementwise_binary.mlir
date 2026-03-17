// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: JAX_ROUNDTRIP
// RUN: JAX_GENERIC_ROUNDTRIP
// RUN: XDSL_JAX_ROUNDTRIP
// RUN: XDSL_JAX_GENERIC_ROUNDTRIP

// CHECK: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
%t0 = "test.op"() : () -> tensor<i32>
// CHECK: %[[TF32:.*]] = "test.op"() : () -> tensor<f32>
// CHECK-GENERIC: %[[TF32:.*]] = "test.op"() : () -> tensor<f32>
%tf32 = "test.op"() : () -> tensor<f32>
// CHECK: %[[TF64:.*]] = "test.op"() : () -> tensor<f64>
// CHECK-GENERIC: %[[TF64:.*]] = "test.op"() : () -> tensor<f64>
%tf64 = "test.op"() : () -> tensor<f64>
// CHECK: %[[T5F32:.*]] = "test.op"() : () -> tensor<5xf32>
// CHECK-GENERIC: %[[T5F32:.*]] = "test.op"() : () -> tensor<5xf32>
%t5f32 = "test.op"() : () -> tensor<5xf32>
// CHECK: %[[TDF32:.*]] = "test.op"() : () -> tensor<?xf32>
// CHECK-GENERIC: %[[TDF32:.*]] = "test.op"() : () -> tensor<?xf32>
%tdf32 = "test.op"() : () -> tensor<?xf32>

// CHECK: %[[ADD:.*]] = stablehlo.add %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[ADD:.*]] = "stablehlo.add"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%add = stablehlo.add %t0, %t0 : tensor<i32>

// CHECK: %[[AND:.*]] = stablehlo.and %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[AND:.*]] = "stablehlo.and"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%and = stablehlo.and %t0, %t0 : tensor<i32>

// CHECK: %[[ATAN2:.*]] = stablehlo.atan2 %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[ATAN2:.*]] = "stablehlo.atan2"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%atan2 = stablehlo.atan2 %tf32, %tf32 : tensor<f32>

// CHECK: %[[COMPLEX:.*]] = stablehlo.complex %[[TF32]], %[[TF32]] : tensor<complex<f32>>
// CHECK-GENERIC: %[[COMPLEX:.*]] = "stablehlo.complex"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
%complex = stablehlo.complex %tf32, %tf32 : tensor<complex<f32>>

// CHECK: %[[COMPLEX2:.*]] = stablehlo.complex %[[T5F32]], %[[T5F32]] : tensor<5xcomplex<f32>>
// CHECK-GENERIC: %[[COMPLEX2:.*]] = "stablehlo.complex"(%[[T5F32]], %[[T5F32]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xcomplex<f32>>
%complex2 = stablehlo.complex %t5f32, %t5f32 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xcomplex<f32>>

// CHECK: [[COMPLEX_FALLBACK:.*]] = stablehlo.complex %[[TDF32:.*]], %[[TDF32:.*]] : (tensor<?xf32>, tensor<?xf32>) -> tensor<5xcomplex<f32>>
// CHECK-GENERIC: [[COMPLEX_FALLBACK:.*]] = "stablehlo.complex"(%[[TDF32:.*]], %[[TDF32:.*]]) : (tensor<?xf32>, tensor<?xf32>) -> tensor<5xcomplex<f32>>
%complex_fallback = stablehlo.complex %tdf32, %tdf32 : (tensor<?xf32>, tensor<?xf32>) -> tensor<5xcomplex<f32>>

// CHECK: %[[DIVIDE:.*]] = stablehlo.divide %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[DIVIDE:.*]] = "stablehlo.divide"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%divide = stablehlo.divide %tf32, %tf32 : tensor<f32>

// CHECK: %[[MAXIMUM:.*]] = stablehlo.maximum %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[MAXIMUM:.*]] = "stablehlo.maximum"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%maximum = stablehlo.maximum %tf32, %tf32 : tensor<f32>

// CHECK: %[[MINIMUM:.*]] = stablehlo.minimum %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[MINIMUM:.*]] = "stablehlo.minimum"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%minimum = stablehlo.minimum %tf32, %tf32 : tensor<f32>

// CHECK: %[[MULTIPLY:.*]] = stablehlo.multiply %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[MULTIPLY:.*]] = "stablehlo.multiply"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%multiply = stablehlo.multiply %t0, %t0 : tensor<i32>

// CHECK: %[[OR:.*]] = stablehlo.or %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[OR:.*]] = "stablehlo.or"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%or = stablehlo.or %t0, %t0 : tensor<i32>

// CHECK: %[[POWER:.*]] = stablehlo.power %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[POWER:.*]] = "stablehlo.power"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%power = stablehlo.power %tf32, %tf32 : tensor<f32>

// CHECK: %[[REMAINDER:.*]] = stablehlo.remainder %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[REMAINDER:.*]] = "stablehlo.remainder"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%remainder = stablehlo.remainder %tf32, %tf32 : tensor<f32>

// CHECK: %[[SHIFT_LEFT:.*]] = stablehlo.shift_left %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SHIFT_LEFT:.*]] = "stablehlo.shift_left"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_left = stablehlo.shift_left %t0, %t0 : tensor<i32>

// CHECK: %[[SHIFT_RIGHT_ARITHMETIC:.*]] = stablehlo.shift_right_arithmetic %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SHIFT_RIGHT_ARITHMETIC:.*]] = "stablehlo.shift_right_arithmetic"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_arithmetic = stablehlo.shift_right_arithmetic %t0, %t0 : tensor<i32>

// CHECK: %[[SHIFT_RIGHT_LOGICAL:.*]] = stablehlo.shift_right_logical %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SHIFT_RIGHT_LOGICAL:.*]] = "stablehlo.shift_right_logical"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_logical = stablehlo.shift_right_logical %t0, %t0 : tensor<i32>

// CHECK: %[[SUBTRACT:.*]] = stablehlo.subtract %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SUBTRACT:.*]] = "stablehlo.subtract"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%subtract = stablehlo.subtract %t0, %t0 : tensor<i32>

// CHECK: %[[XOR:.*]] = stablehlo.xor %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[XOR:.*]] = "stablehlo.xor"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%xor = stablehlo.xor %t0, %t0 : tensor<i32>
