// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

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

// CHECK: lhs component count must be positive
"test.op"() {algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 0, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>} : () -> ()

// -----

// CHECK: rhs component count must be positive
"test.op"() {algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 0, num_primitive_operations = 1, allow_imprecise_accumulation = false>} : () -> ()

// -----

// CHECK: num primitive operations must be positive
"test.op"() {algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 0, allow_imprecise_accumulation = false>} : () -> ()

// -----

%dot_lhs3 = "test.op"() : () -> tensor<2x3xi32>
%dot_rhs3 = "test.op"() : () -> tensor<3x4xi32>
// CHECK: Operation does not verify: must specify DEFAULT precision config when algorithm is set.
%dot3 = stablehlo.dot_general %dot_lhs3, %dot_rhs3, batching_dims = [] x [], contracting_dims = [1] x [0], precision = [DEFAULT, HIGH], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>

// -----

%dot_lhs4 = "test.op"() : () -> tensor<2x3xi32>
%dot_rhs4 = "test.op"() : () -> tensor<3x4xi32>
// CHECK: Operation does not verify: expects precision config to be empty or have <= 2 elements.
%dot4 = stablehlo.dot_general %dot_lhs4, %dot_rhs4, batching_dims = [] x [], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT, HIGH] : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>

// -----

%dot_lhs5 = "test.op"() : () -> tensor<2x3x4xi32>
%dot_rhs5 = "test.op"() : () -> tensor<2x4x5xi32>
// CHECK: Operation does not verify: lhs and rhs should have the same number of batching dimensions
%dot5 = stablehlo.dot_general %dot_lhs5, %dot_rhs5, batching_dims = [0] x [], contracting_dims = [2] x [1] : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>

// -----

%dot_lhs6 = "test.op"() : () -> tensor<2x3x4xi32>
%dot_rhs6 = "test.op"() : () -> tensor<2x4x5xi32>
// CHECK: Operation does not verify: lhs and rhs should have the same number of contracting dimensions
%dot6 = stablehlo.dot_general %dot_lhs6, %dot_rhs6, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1] : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x5xi32>

// -----

%dot_lhs7 = "test.op"() : () -> tensor<2x3x4xi32>
%dot_rhs7 = "test.op"() : () -> tensor<2x4x5xi32>
// CHECK: Operation does not verify: has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 0
%dot7 = stablehlo.dot_general %dot_lhs7, %dot_rhs7, batching_dims = [0] x [0], contracting_dims = [0] x [1] : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<3x5xi32>

// -----

%dot_lhs8 = "test.op"() : () -> tensor<2x3x4xi32>
%dot_rhs8 = "test.op"() : () -> tensor<2x4x5xi32>
// CHECK: Operation does not verify: lhs_batching_dimensions value: 3 is out of range: [0, 3)
%dot8 = stablehlo.dot_general %dot_lhs8, %dot_rhs8, batching_dims = [3] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<3x5xi32>

// -----

%dot_lhs9 = "test.op"() : () -> tensor<2x3x4xi32>
%dot_rhs9 = "test.op"() : () -> tensor<5x4x6xi32>
// CHECK: Operation does not verify: batching dimension sizes must match for lhs/rhs
%dot9 = stablehlo.dot_general %dot_lhs9, %dot_rhs9, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xi32>, tensor<5x4x6xi32>) -> tensor<2x3x6xi32>

// -----

%dot_lhs10 = "test.op"() : () -> tensor<2x3x4xi32>
%dot_rhs10 = "test.op"() : () -> tensor<2x5x6xi32>
// CHECK: Operation does not verify: contracting dimension sizes must match for lhs/rhs
%dot10 = stablehlo.dot_general %dot_lhs10, %dot_rhs10, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xi32>, tensor<2x5x6xi32>) -> tensor<2x3x6xi32>

// -----

%dot_lhs11 = "test.op"() : () -> tensor<2x3x4xi32>
%dot_rhs11 = "test.op"() : () -> tensor<2x4x6xi32>
// CHECK: Operation does not verify: inferred shape '(2, 3, 6)' is incompatible with return type of operation tensor<2x6xi32>
%dot11 = stablehlo.dot_general %dot_lhs11, %dot_rhs11, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x3x4xi32>, tensor<2x4x6xi32>) -> tensor<2x6xi32>

// -----

%dot_lhs_bad_precision = "test.op"() : () -> tensor<2x3xi32>
%dot_rhs_bad_precision = "test.op"() : () -> tensor<3x4xi32>
// CHECK: unknown precision enum 'FOO'
%dot_bad_precision = stablehlo.dot_general %dot_lhs_bad_precision, %dot_rhs_bad_precision, batching_dims = [] x [], contracting_dims = [1] x [0], precision = [FOO] : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
