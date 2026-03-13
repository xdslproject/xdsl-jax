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

// -----

%bcast_operand = "test.op"() : () -> tensor<1x3xi32>
// CHECK: broadcast_dimensions size (1) does not match operand rank (2)
%bad_bcast_size = stablehlo.broadcast_in_dim %bcast_operand, dims = [2] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>

// -----

%bcast_operand2 = "test.op"() : () -> tensor<1x3xi32>
// CHECK: broadcast_dimensions should not have duplicates
%bad_bcast_dups = stablehlo.broadcast_in_dim %bcast_operand2, dims = [2, 2] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>

// -----

%bcast_operand3 = "test.op"() : () -> tensor<1x3xi32>
// CHECK: broadcast_dimensions contains invalid value 5 for result with rank 3
%bad_bcast_invalid = stablehlo.broadcast_in_dim %bcast_operand3, dims = [2, 5] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>

// -----

%bcast_operand4 = "test.op"() : () -> tensor<2x3xi32>
// CHECK: size of operand dimension 1 (3) is not equal to 1 or size of result dimension 1 (2)
%bad_bcast_dim_size = stablehlo.broadcast_in_dim %bcast_operand4, dims = [0, 1] : (tensor<2x3xi32>) -> tensor<2x2x2xi32>

// -----

%bcast_operand5 = "test.op"() : () -> tensor<1x3xi32>
// CHECK: broadcast_in_dim output must have a static shape.
%bad_bcast_dynamic_result = stablehlo.broadcast_in_dim %bcast_operand5, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<?x3x2xi32>

// -----

%arg = "test.op"() : () -> tensor<2x3xi32>
// CHECK: Operation does not verify: Layout attributes should be specified for either both operands and results or none.
%custom_call_missing_layouts = stablehlo.custom_call @foo(%arg) {
  operand_layouts = [dense<[0, 1]> : tensor<2xindex>],
  output_operand_aliases = []
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: Number of results must match the number of result layouts
%arg1 = "test.op"() : () -> tensor<2x3xi32>
%custom_call_layout_count_results = stablehlo.custom_call @foo(%arg1) {
  operand_layouts = [dense<[0, 1]> : tensor<2xindex>],
  result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>],
  output_operand_aliases = []
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: Number of operands must match the number of operand layouts
%arg0 = "test.op"() : () -> tensor<2x3xi32>
%custom_call_layout_count_operands = stablehlo.custom_call @foo(%arg0) {
  operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>],
  result_layouts = [dense<[0, 1]> : tensor<2xindex>],
  output_operand_aliases = []
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: Only tensor types can have non-empty layout
%arg_tensor = "test.op"() : () -> tensor<2x3xi32>
%arg_token = "test.op"() : () -> !stablehlo.token
%custom_call_token_layout = stablehlo.custom_call @foo(%arg_tensor, %arg_token) {
  operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>],
  result_layouts = [dense<[0, 1]> : tensor<2xindex>],
  output_operand_aliases = []
} : (tensor<2x3xi32>, !stablehlo.token) -> tensor<2x3xi32>

// -----

%arg = "test.op"() : () -> tensor<2x3xi32>
// CHECK: Operation does not verify: incorrect layout [0, 0] for type tensor<2x3xi32>, layout must be a permutation of [0, 2)
%custom_call_bad_layout = stablehlo.custom_call @foo(%arg) {
  operand_layouts = [dense<[0, 0]> : tensor<2xindex>],
  result_layouts = [dense<[0, 1]> : tensor<2xindex>],
  output_operand_aliases = []
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: Tuple types are not fully supported with layout constraints yet
%tuple_arg = "test.op"() : () -> tuple<tensor<2x3xi32>>
%custom_call_tuple_layout = stablehlo.custom_call @foo(%tuple_arg) {
  operand_layouts = [dense<[0, 1]> : tensor<2xindex>],
  result_layouts = [dense<[0, 1]> : tensor<2xindex>],
  output_operand_aliases = []
} : (tuple<tensor<2x3xi32>>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: output_tuple_indices in the output_operand_alias attribute out of bounds
%arg_tuple_result = "test.op"() : () -> tensor<2x3xi32>
%custom_call_alias_tuple_output_bounds = stablehlo.custom_call @foo(%arg_tuple_result) {
  output_operand_aliases = [
    #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>
  ]
} : (tensor<2x3xi32>) -> tuple<tensor<2x3xi32>>

// -----

// CHECK: Operation does not verify: output_tuple_indices in the output_operand_alias attribute out of bounds
%arg2 = "test.op"() : () -> tensor<2x3xi32>
%custom_call_alias_output_bounds = stablehlo.custom_call @foo(%arg2) {
  output_operand_aliases = [
    #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>
  ]
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: operand_tuple_indices in the output_operand_alias attribute out of bounds
%arg3 = "test.op"() : () -> tensor<2x3xi32>
%custom_call_alias_operand_bounds = stablehlo.custom_call @foo(%arg3) {
  output_operand_aliases = [
    #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = [0]>
  ]
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

%arg = "test.op"() : () -> tensor<2x3xi32>
// CHECK: Operation does not verify: expects operandIndex in the output_operand_alias attribute to be in range [0, 1); got: 1.
%custom_call_bad_alias = stablehlo.custom_call @foo(%arg) {
  output_operand_aliases = [
    #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = [0]>
  ]
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: shapes mismatch in the output_operand_alias attribute
%arg4 = "test.op"() : () -> tensor<2x3xi32>
%custom_call_alias_shape_mismatch = stablehlo.custom_call @foo(%arg4) {
  output_operand_aliases = [
    #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
  ]
} : (tensor<2x3xi32>) -> tensor<2x4xi32>

// -----

%arg = "test.op"() : () -> tensor<2x3xi32>
// CHECK: Operation does not verify: invalid api_version value 6
%custom_call_bad_api_version = stablehlo.custom_call @foo(%arg) {
  api_version = 6 : i32
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

%arg = "test.op"() : () -> tensor<2x3xi32>
// CHECK: Operation does not verify: backend_config for api_version API_VERSION_TYPED_FFI must be a dictionary attribute.
%custom_call_bad_backend_config = stablehlo.custom_call @foo(%arg) {
  api_version = 4 : i32,
  backend_config = "not a dict",
  output_operand_aliases = []
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: backend_config for api_version API_VERSION_ORIGINAL must be a string attribute.
%arg5 = "test.op"() : () -> tensor<2x3xi32>
%custom_call_backend_config_string = stablehlo.custom_call @foo(%arg5) {
  api_version = 1 : i32,
  backend_config = {foo = 42 : i32},
  output_operand_aliases = []
} : (tensor<2x3xi32>) -> tensor<2x3xi32>

// -----

// CHECK: Operation does not verify: Iota output must have a static shape.
%iota = stablehlo.iota dim = 0 : tensor<?x3xi32>

// -----

// CHECK: Operation does not verify: Iota does not support scalars.
%iota = stablehlo.iota dim = 0 : tensor<i32>

// -----

// CHECK: Operation does not verify: Iota dimension cannot go beyond the output rank.
%iota = stablehlo.iota dim = 3 : tensor<2x3xi32>

// -----

%slice_input = "test.op"() : () -> tensor<3x8xi64>
// CHECK: Operation does not verify: all of {start_indices, limit_indices, strides} must have the same size
%slice = "stablehlo.slice"(%slice_input) <{start_indices = array<i64: 1, 4>, limit_indices = array<i64: 3>, strides = array<i64: 1, 2>}> : (tensor<3x8xi64>) -> tensor<2x2xi64>

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect operands to be compatible with condition block arguments
%while_bad_cond_args = "stablehlo.while"(%while_init) ({
^bb0(%arg0: tensor<i32>):
  %pred = "test.op"() : () -> tensor<i1>
  "stablehlo.return"(%pred) : (tensor<i1>) -> ()
}, {
^bb0(%arg0: tensor<i64>):
  "stablehlo.return"(%arg0) : (tensor<i64>) -> ()
}) : (tensor<i64>) -> tensor<i64>

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect operands to be compatible with body block arguments
%while_bad_body_args = "stablehlo.while"(%while_init) ({
^bb0(%arg0: tensor<i64>):
  %pred = "test.op"() : () -> tensor<i1>
  "stablehlo.return"(%pred) : (tensor<i1>) -> ()
}, {
^bb0(%arg0: tensor<i32>):
  %body = "test.op"() : () -> tensor<i64>
  "stablehlo.return"(%body) : (tensor<i64>) -> ()
}) : (tensor<i64>) -> tensor<i64>

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect operands to be compatible with body block return types
%while_bad_body_return = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %pred = "test.op"() : () -> tensor<i1>
  stablehlo.return %pred : tensor<i1>
} do {
  %body = "test.op"() : () -> tensor<i32>
  stablehlo.return %body : tensor<i32>
}

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect condition body returns a single value but got 2
%while_bad_cond_return_count = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %pred = "test.op"() : () -> tensor<i1>
  stablehlo.return %pred, %pred : tensor<i1>, tensor<i1>
} do {
  stablehlo.return %arg0 : tensor<i64>
}

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect condition block return a zero-ranked tensor of i1 but got tensor<1xi1>
%while_bad_cond_return_type = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %pred = "test.op"() : () -> tensor<1xi1>
  stablehlo.return %pred : tensor<1xi1>
} do {
  stablehlo.return %arg0 : tensor<i64>
}

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect condition block return a zero-ranked tensor of i1 but got !stablehlo.token
%while_bad_cond_return_token = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %tok = "test.op"() : () -> !stablehlo.token
  stablehlo.return %tok : !stablehlo.token
} do {
  stablehlo.return %arg0 : tensor<i64>
}

// -----

%arg = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: requires the same number of operands and results (got 1 operands and 2 results)
%bad0, %bad1 = "stablehlo.optimization_barrier"(%arg) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)

// -----

%arg = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: requires the same type for operand and result at index 0 (got tensor<i32> vs tensor<i1>)
%bad = "stablehlo.optimization_barrier"(%arg) : (tensor<i32>) -> tensor<i1>

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

// -----

%pad_operand = "test.op"() : () -> tensor<2x3xi32>
%pad_value_rank1 = "test.op"() : () -> tensor<1xi32>
// CHECK: Operation does not verify: Expect padding_value is an 0-dimensional tensor
%bad_pad_value_rank = stablehlo.pad %pad_operand, %pad_value_rank1,
  low = [0, 1],
  high = [2, 1],
  interior = [1, 2] : (tensor<2x3xi32>, tensor<1xi32>) -> tensor<5x9xi32>

// -----

%pad_operand_rank = "test.op"() : () -> tensor<2x3xi32>
%pad_value_rank = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Pad operation rank mismatch while the operand has 2 dimension(s) and result shape has 1 dimension(s)
%bad_pad_rank = stablehlo.pad %pad_operand_rank, %pad_value_rank,
  low = [0, 1],
  high = [2, 1],
  interior = [1, 2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5xi32>

// -----

%pad_operand_negative = "test.op"() : () -> tensor<2x3xi32>
%pad_value_negative = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: The interior_padding value must be equal or larger than 0, found -1
%bad_pad_negative_interior = stablehlo.pad %pad_operand_negative, %pad_value_negative,
  low = [0, 1],
  high = [2, 1],
  interior = [1, -1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x1xi32>

// -----

%pad_operand_shape = "test.op"() : () -> tensor<2x3xi32>
%pad_value_shape = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: Pad operation at 1 dimension  mismatch
%bad_pad_shape = stablehlo.pad %pad_operand_shape, %pad_value_shape,
  low = [0, 1],
  high = [2, 1],
  interior = [1, 2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x8xi32>
