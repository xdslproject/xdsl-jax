// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%operand = "test.op"() : () -> tensor<2x3x2xi32>

%result = "stablehlo.transpose"(%operand) {
  permutation = array<i64: 5, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>

// CHECK: Permutation (5, 1, 0) of transpose must be a permutation of range(3)

// -----

%operand = "test.op"() : () -> tensor<2x3x2xi32>

%result = "stablehlo.transpose"(%operand) {
  permutation = array<i64: 2, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<4x3x2xi32>

// CHECK: Operation does not verify: Permutation mismatch at dimension 0, expected 2

// -----

%operand = "test.op"() : () -> tensor<i32>

// CHECK: Operation does not verify: operand 'lhs' at position 0 does not verify:
// CHECK: Unexpected attribute i32
%result = "stablehlo.atan2"(%operand, %operand) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// -----

%operand = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: operand 'operand' at position 0 does not verify:
// CHECK: Unexpected attribute i32
%result = "stablehlo.cbrt"(%operand) : (tensor<i32>) -> tensor<i32>

// -----

%operand = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: operand 'operand' at position 0 does not verify:
// CHECK: Unexpected attribute i32
%result = "stablehlo.ceil"(%operand) : (tensor<i32>) -> tensor<i32>

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

// CHECK: unknown field 'unknown_field'
"test.op"() {gather = #stablehlo.gather<
  offset_dims = [0],
  unknown_field = [1]
>} : () -> ()

// -----

// CHECK: duplicate 'offset_dims' field
"test.op"() {gather = #stablehlo.gather<
  offset_dims = [0],
  offset_dims = [1]
>} : () -> ()

// -----

// CHECK: Operation does not verify: Reduce op expects at least one input/init_value pair
"stablehlo.reduce"() ({
  "stablehlo.return"() : () -> ()
}) {dimensions = array<i64: 0>} : () -> ()

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce op requires the same number of inputs, init_values, and results
%reduce0, %reduce1 = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> (tensor<2xi32>, tensor<2xi32>)
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : tensor<i32>
}

// -----

%input0 = "test.op"() : () -> tensor<2x3xi32>
%input1 = "test.op"() : () -> tensor<2x4xi32>
%init0 = "test.op"() : () -> tensor<i32>
%init1 = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce inputs must have the same shape.
%reduce_shape0, %reduce_shape1 = stablehlo.reduce (%input0 init: %init0), (%input1 init: %init1) across dimensions = [1] : (tensor<2x3xi32>, tensor<2x4xi32>, tensor<i32>, tensor<i32>) -> (tensor<2xi32>, tensor<2xi32>)
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) (%arg2 : tensor<i32>, %arg3 : tensor<i32>) {
  stablehlo.return %arg0, %arg2 : tensor<i32>, tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce dimensions must be unique, got (1, 1)
%reduce_dims_unique = stablehlo.reduce (%input init: %init) across dimensions = [1, 1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce dimension 2 out of range for rank 2
%reduce_dim_range = stablehlo.reduce (%input init: %init) across dimensions = [2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<1xi32>
// CHECK: Operation does not verify: Reduce init_values must be 0-dimensional tensors; found rank 1 at index 0
%reduce_init_rank = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<1xi32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: Reduce input and init_value element types must match at index 0
%reduce_init_type = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i64>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce result shape mismatch at index 0: expected (2,), got (3,)
%reduce_result_shape = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce result element types must match input element types at index 0
%reduce_result_type = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi64>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body must take 2 arguments, got 1
%reduce_body_args = "stablehlo.reduce"(%input, %init) ({
^bb0(%arg0 : tensor<i32>):
  "stablehlo.return"(%arg0) : (tensor<i32>) -> ()
}) {dimensions = array<i64: 1>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body arguments for pair 0 must be tensor types
%reduce_body_arg_type = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : !stablehlo.token, %arg1 : tensor<i32>) {
  stablehlo.return %arg0 : !stablehlo.token
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body arguments for pair 0 must be 0-dim tensors
%reduce_body_arg_rank = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<1xi32>, %arg1 : tensor<1xi32>) {
  stablehlo.return %arg0 : tensor<1xi32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body argument element types must match input element types at index 0
%reduce_body_arg_elem = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i64>, %arg1 : tensor<i64>) {
  stablehlo.return %arg0 : tensor<i64>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body must return 1 values, got 2
%reduce_body_return_count = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  stablehlo.return %arg0, %arg0 : tensor<i32>, tensor<i32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body return value at index 0 must be a tensor
%reduce_body_return_type = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  %tok = "test.op"() : () -> !stablehlo.token
  stablehlo.return %tok : !stablehlo.token
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body return value at index 0 must be 0-dim tensor
%reduce_body_return_rank = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  %val = "test.op"() : () -> tensor<1xi32>
  stablehlo.return %val : tensor<1xi32>
}

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
%init = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Reduce body return element types must match input element types at index 0
%reduce_body_return_elem = stablehlo.reduce (%input init: %init) across dimensions = [1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi32>
reducer (%arg0 : tensor<i32>, %arg1 : tensor<i32>) {
  %val = "test.op"() : () -> tensor<i64>
  stablehlo.return %val : tensor<i64>
}

// -----

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

// CHECK: Operation does not verify: Iota output must have a static shape.
%iota = stablehlo.iota dim = 0 : tensor<?x3xi32>

// -----

// CHECK: Operation does not verify: Iota does not support scalars.
%iota = stablehlo.iota dim = 0 : tensor<i32>

// -----

// CHECK: Operation does not verify: Iota dimension cannot go beyond the output rank.
%iota = stablehlo.iota dim = 3 : tensor<2x3xi32>
