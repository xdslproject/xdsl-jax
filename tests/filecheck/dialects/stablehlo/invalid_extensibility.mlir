// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

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
