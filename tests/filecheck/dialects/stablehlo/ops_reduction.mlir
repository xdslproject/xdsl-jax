// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: JAX_ROUNDTRIP
// RUN: JAX_GENERIC_ROUNDTRIP
// RUN: XDSL_JAX_ROUNDTRIP
// RUN: XDSL_JAX_GENERIC_ROUNDTRIP

// CHECK: %[[REDUCE_INPUT0:[^ ]+]] = "test.op"() : () -> tensor<2x3xi64>
// CHECK-GENERIC: %[[REDUCE_INPUT0:[^ ]+]] = "test.op"() : () -> tensor<2x3xi64>
%reduce_input0 = "test.op"() : () -> tensor<2x3xi64>
// CHECK: %[[REDUCE_INPUT1:[^ ]+]] = "test.op"() : () -> tensor<2x3xi64>
// CHECK-GENERIC: %[[REDUCE_INPUT1:[^ ]+]] = "test.op"() : () -> tensor<2x3xi64>
%reduce_input1 = "test.op"() : () -> tensor<2x3xi64>
// CHECK: %[[REDUCE_INIT0:[^ ]+]] = "test.op"() : () -> tensor<i64>
// CHECK-GENERIC: %[[REDUCE_INIT0:[^ ]+]] = "test.op"() : () -> tensor<i64>
%reduce_init0 = "test.op"() : () -> tensor<i64>
// CHECK: %[[REDUCE_INIT1:[^ ]+]] = "test.op"() : () -> tensor<i64>
// CHECK-GENERIC: %[[REDUCE_INIT1:[^ ]+]] = "test.op"() : () -> tensor<i64>
%reduce_init1 = "test.op"() : () -> tensor<i64>

// CHECK: %[[REDUCE_MULTI:[^ ]+]] = stablehlo.reduce(%[[REDUCE_INPUT0:[^ )]+]] init: %[[REDUCE_INIT0:[^ )]+]]), (%[[REDUCE_INPUT1:[^ )]+]] init: %[[REDUCE_INIT1:[^ )]+]])
// CHECK-SAME: across dimensions = [1] : (tensor<2x3xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>)
// CHECK-NEXT: reducer{{ ?}}(%[[REDUCE_ARG0:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_ARG2:.*]]{{ ?}}: tensor<i64>) (%[[REDUCE_ARG1:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_ARG3:.*]]{{ ?}}: tensor<i64>) {
// CHECK:   stablehlo.return %[[REDUCE_RET0:.*]], %[[REDUCE_RET1:.*]] : tensor<i64>, tensor<i64>
// CHECK: }
// CHECK-GENERIC: %[[REDUCE_MULTI:[^ ]+]] = "stablehlo.reduce"(%[[REDUCE_INPUT0]], %[[REDUCE_INPUT1]], %[[REDUCE_INIT0]], %[[REDUCE_INIT1]]) <{dimensions = array<i64: 1>}> ({
// CHECK-GENERIC:   ^bb[[REDUCE_MULTI_BB:[0-9]+]](%[[REDUCE_GEN_ARG0:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_GEN_ARG1:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_GEN_ARG2:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_GEN_ARG3:.*]]{{ ?}}: tensor<i64>):
// CHECK-GENERIC:     "stablehlo.return"(%[[REDUCE_GEN_RET0:.*]], %[[REDUCE_GEN_RET1:.*]]) : (tensor<i64>, tensor<i64>) -> ()
// CHECK-GENERIC: }) : (tensor<2x3xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>)
%reduce_multi_0, %reduce_multi_1 = stablehlo.reduce (%reduce_input0 init: %reduce_init0), (%reduce_input1 init: %reduce_init1) across dimensions = [1] : (tensor<2x3xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> (tensor<2xi64>, tensor<2xi64>)
reducer (%reduce_arg0 : tensor<i64>, %reduce_arg1 : tensor<i64>) (%reduce_arg2 : tensor<i64>, %reduce_arg3 : tensor<i64>) {
  stablehlo.return %reduce_arg0, %reduce_arg2 : tensor<i64>, tensor<i64>
}

%dot_lhs = "test.op"() : () -> tensor<2x3xi32>
%dot_rhs = "test.op"() : () -> tensor<3x4xi32>
// CHECK: %dot_no_algorithm = stablehlo.dot_general %dot_lhs, %dot_rhs, contracting_dims = [1] x [0]: (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
// CHECK-GENERIC: %dot_no_algorithm = "stablehlo.dot_general"(%dot_lhs, %dot_rhs) <{dot_dimension_numbers = #stablehlo.dot<
// CHECK-GENERIC-SAME: lhs_contracting_dimensions = [1],
// CHECK-GENERIC-SAME: rhs_contracting_dimensions = [0]
// CHECK-GENERIC-SAME: >}> : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
%dot_no_algorithm = stablehlo.dot_general %dot_lhs, %dot_rhs, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>

// CHECK: %dot = stablehlo.dot_general %dot_lhs, %dot_rhs, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <
// CHECK-NEXT: lhs_precision_type = f32,
// CHECK-NEXT: rhs_precision_type = f32,
// CHECK-NEXT: accumulation_type = f32,
// CHECK-NEXT: lhs_component_count = 1,
// CHECK-NEXT: rhs_component_count = 1,
// CHECK-NEXT: num_primitive_operations = 1,
// CHECK-NEXT: allow_imprecise_accumulation = false
// CHECK-NEXT: >: (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
// CHECK-GENERIC: %dot = "stablehlo.dot_general"(%dot_lhs, %dot_rhs) <{dot_dimension_numbers = #stablehlo.dot<
// CHECK-GENERIC-SAME: lhs_contracting_dimensions = [1],
// CHECK-GENERIC-SAME: rhs_contracting_dimensions = [0]
// CHECK-GENERIC-SAME: >, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], algorithm = #stablehlo.dot_algorithm<
// CHECK-GENERIC-NEXT: lhs_precision_type = f32,
// CHECK-GENERIC-NEXT: rhs_precision_type = f32,
// CHECK-GENERIC-NEXT: accumulation_type = f32,
// CHECK-GENERIC-NEXT: lhs_component_count = 1,
// CHECK-GENERIC-NEXT: rhs_component_count = 1,
// CHECK-GENERIC-NEXT: num_primitive_operations = 1,
// CHECK-GENERIC-NEXT: allow_imprecise_accumulation = false
// CHECK-GENERIC-NEXT: >}> : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>
%dot = stablehlo.dot_general %dot_lhs, %dot_rhs, batching_dims = [] x [], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<2x3xi32>, tensor<3x4xi32>) -> tensor<2x4xi32>

%dot_batch_lhs = "test.op"() : () -> tensor<2x3x4xi32>
%dot_batch_rhs = "test.op"() : () -> tensor<2x4x5xi32>
// CHECK: %dot_with_batching_and_algorithm = stablehlo.dot_general %dot_batch_lhs, %dot_batch_rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1], algorithm = <
// CHECK-NEXT: lhs_precision_type = f32,
// CHECK-NEXT: rhs_precision_type = f32,
// CHECK-NEXT: accumulation_type = f32,
// CHECK-NEXT: lhs_component_count = 1,
// CHECK-NEXT: rhs_component_count = 1,
// CHECK-NEXT: num_primitive_operations = 1,
// CHECK-NEXT: allow_imprecise_accumulation = false
// CHECK-NEXT: >: (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>
// CHECK-GENERIC: %dot_with_batching_and_algorithm = "stablehlo.dot_general"(%dot_batch_lhs, %dot_batch_rhs) <{dot_dimension_numbers = #stablehlo.dot<
// CHECK-GENERIC-SAME: lhs_batching_dimensions = [0],
// CHECK-GENERIC-SAME: rhs_batching_dimensions = [0],
// CHECK-GENERIC-SAME: lhs_contracting_dimensions = [2],
// CHECK-GENERIC-SAME: rhs_contracting_dimensions = [1]
// CHECK-GENERIC-SAME: >, algorithm = #stablehlo.dot_algorithm<
// CHECK-GENERIC-NEXT: lhs_precision_type = f32,
// CHECK-GENERIC-NEXT: rhs_precision_type = f32,
// CHECK-GENERIC-NEXT: accumulation_type = f32,
// CHECK-GENERIC-NEXT: lhs_component_count = 1,
// CHECK-GENERIC-NEXT: rhs_component_count = 1,
// CHECK-GENERIC-NEXT: num_primitive_operations = 1,
// CHECK-GENERIC-NEXT: allow_imprecise_accumulation = false
// CHECK-GENERIC-NEXT: >}> : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>
%dot_with_batching_and_algorithm = stablehlo.dot_general %dot_batch_lhs, %dot_batch_rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>
