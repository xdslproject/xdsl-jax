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
