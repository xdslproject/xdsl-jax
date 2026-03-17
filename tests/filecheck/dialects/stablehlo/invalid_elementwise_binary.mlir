// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%operand = "test.op"() : () -> tensor<i32>

// CHECK: Operation does not verify: operand 'lhs' at position 0 does not verify:
// CHECK: Unexpected attribute i32
%result = "stablehlo.atan2"(%operand, %operand) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// -----

%operand = "test.op"() : () -> tensor<5xf16>
// CHECK: Operation does not verify: operand 'lhs' at position 0 does not verify:
// CHECK: Unexpected attribute f16
%result = "stablehlo.complex"(%operand, %operand) : (tensor<5xf16>, tensor<5xf16>) -> tensor<5xcomplex<f16>>

// -----

%operand = "test.op"() : () -> tensor<5xf32>
// CHECK: expected tensor with complex element type
%result = stablehlo.complex %operand, %operand : tensor<5xf32>

// -----

%operand = "test.op"() : () -> tensor<f32>
// CHECK: Operation does not verify: 'stablehlo.complex' op inferred type(s) 'tensor<complex<f32>>' are incompatible with return type(s) of operation 'tensor<complex<f64>>'
%result = stablehlo.complex %operand, %operand : (tensor<f32>, tensor<f32>) -> tensor<complex<f64>>

// -----

%operand = "test.op"() : () -> tensor<2xf32>
// CHECK: Operation does not verify: 'stablehlo.complex' requires the same shape for all operands and results
%result = stablehlo.complex %operand, %operand : (tensor<2xf32>, tensor<2xf32>) -> tensor<3xcomplex<f32>>
