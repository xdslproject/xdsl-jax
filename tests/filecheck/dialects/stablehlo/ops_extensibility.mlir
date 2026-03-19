// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: %[[ARG:.*]] = "test.op"() : () -> tensor<2x2xf32>
// CHECK-GENERIC: %[[ARG:.*]] = "test.op"() : () -> tensor<2x2xf32>
%arg = "test.op"() : () -> tensor<2x2xf32>

// CHECK: %[[CUSTOM_CALL_LAYOUTS:.*]] = stablehlo.custom_call @bar(%[[ARG]]) {
// CHECK-SAME: api_version = 4 : i32,
// CHECK-SAME: backend_config = {bar = 42 : i32},
// CHECK-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
// CHECK-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-GENERIC: "stablehlo.custom_call"(%[[ARG]])
// CHECK-GENERIC-DAG: api_version = 4 : i32
// CHECK-GENERIC-DAG: backend_config = {bar = 42 : i32}
// CHECK-GENERIC-DAG: call_target_name = "bar"
// CHECK-GENERIC-DAG: has_side_effect = false
// CHECK-GENERIC-DAG: operand_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CHECK-GENERIC-DAG: output_operand_aliases = []
// CHECK-GENERIC-DAG: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CHECK-GENERIC: : (tensor<2x2xf32>) -> tensor<2x2xf32>
%custom_call_layouts = stablehlo.custom_call @bar(%arg) {
  api_version = 4 : i32,
  backend_config = {bar = 42 : i32},
  operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
  result_layouts = [dense<[1, 0]> : tensor<2xindex>],
  output_operand_aliases = [],
  has_side_effect = false
} : (tensor<2x2xf32>) -> tensor<2x2xf32>

// CHECK: %[[TOKEN_INPUT:.*]] = "test.op"() : () -> !stablehlo.token
// CHECK-GENERIC: %[[TOKEN_INPUT:.*]] = "test.op"() : () -> !stablehlo.token
%token_input = "test.op"() : () -> !stablehlo.token
// CHECK: %[[CUSTOM_CALL_TOKEN:.*]] = stablehlo.custom_call @token_passthrough(%[[TOKEN_INPUT]]) {
// CHECK-SAME: backend_config = "opaque-config",
// CHECK-SAME: operand_layouts = [dense<> : tensor<0xindex>],
// CHECK-SAME: result_layouts = [dense<> : tensor<0xindex>]} : (!stablehlo.token) -> !stablehlo.token
// CHECK-GENERIC: "stablehlo.custom_call"(%[[TOKEN_INPUT]])
// CHECK-GENERIC-DAG: backend_config = "opaque-config"
// CHECK-GENERIC-DAG: call_target_name = "token_passthrough"
// CHECK-GENERIC-DAG: operand_layouts = [dense<> : tensor<0xindex>]
// CHECK-GENERIC-DAG: output_operand_aliases = []
// CHECK-GENERIC-DAG: result_layouts = [dense<> : tensor<0xindex>]
// CHECK-GENERIC: : (!stablehlo.token) -> !stablehlo.token
%custom_call_token_layout = stablehlo.custom_call @token_passthrough(%token_input) {
  backend_config = "opaque-config",
  operand_layouts = [dense<> : tensor<0xindex>],
  result_layouts = [dense<> : tensor<0xindex>],
  output_operand_aliases = []
} : (!stablehlo.token) -> !stablehlo.token

// CHECK: %[[CUSTOM_CALL_TUPLE:.*]] = stablehlo.custom_call @tuple_result(%[[ARG]]) {
// CHECK-SAME: api_version = 4 : i32,
// CHECK-SAME: backend_config = {bar = 42 : i32},
// CHECK-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
// CHECK-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> tuple<tensor<2x2xf32>, tensor<2x2xf32>>
// CHECK-GENERIC: "stablehlo.custom_call"(%[[ARG]])
// CHECK-GENERIC-DAG: api_version = 4 : i32
// CHECK-GENERIC-DAG: backend_config = {bar = 42 : i32}
// CHECK-GENERIC-DAG: call_target_name = "tuple_result"
// CHECK-GENERIC-DAG: has_side_effect = false
// CHECK-GENERIC-DAG: operand_layouts = [dense<[1, 0]> : tensor<2xindex>]
// CHECK-GENERIC-DAG: output_operand_aliases = []
// CHECK-GENERIC-DAG: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>]
// CHECK-GENERIC: : (tensor<2x2xf32>) -> tuple<tensor<2x2xf32>, tensor<2x2xf32>>
%custom_call_tuple_result_layouts = stablehlo.custom_call @tuple_result(%arg) {
  api_version = 4 : i32,
  backend_config = {bar = 42 : i32},
  operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
  result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>],
  output_operand_aliases = [],
  has_side_effect = false
} : (tensor<2x2xf32>) -> tuple<tensor<2x2xf32>, tensor<2x2xf32>>

// CHECK: %[[CC_ALIAS:.*]] = "test.op"() : () -> tuple<tensor<1x1xf32>, tensor<2x3xf32>>
%custom_call_alias = "test.op"() : () -> tuple<tensor<1x1xf32>, tensor<2x3xf32>>
// CHECK: %[[CC_ALIAS_1:.*]] = "test.op"() : () -> tensor<5x5xf32>
%custom_call_alias_1 = "test.op"() : () -> tensor<5x5xf32>
// CHECK: %[[CC_ALIAS_RES:.*]] = stablehlo.custom_call @foo(%[[CC_ALIAS]], %[[CC_ALIAS_1]]) {output_operand_aliases = [#stablehlo.output_operand_alias<
// CHECK-NEXT:   output_tuple_indices = [0],
// CHECK-NEXT:   operand_index = 0,
// CHECK-NEXT:   operand_tuple_indices = [1]
// CHECK-NEXT: >]} : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> (tensor<2x3xf32>, tensor<5x5xf32>)
// CHECK-GENERIC: %[[CC_ALIAS_GEN_RES:.*]] = "stablehlo.custom_call"(%[[CC_ALIAS_GEN:.*]], %[[CC_ALIAS_1_GEN:.*]]) <{call_target_name = "foo", output_operand_aliases = [#stablehlo.output_operand_alias<
// CHECK-GENERIC-NEXT:     output_tuple_indices = [0],
// CHECK-GENERIC-NEXT:     operand_index = 0,
// CHECK-GENERIC-NEXT:     operand_tuple_indices = [1]
// CHECK-GENERIC-NEXT:   >], has_side_effect = false, api_version = 1 : i32}> : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> (tensor<2x3xf32>, tensor<5x5xf32>)
%custom_call_result_0, %custom_call_result_1 = stablehlo.custom_call @foo(%custom_call_alias, %custom_call_alias_1) {
  output_operand_aliases = [
    #stablehlo.output_operand_alias<output_tuple_indices = [0],
                               operand_index = 0,
                               operand_tuple_indices = [1]>
  ]
} : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> (tensor<2x3xf32>, tensor<5x5xf32>)
