// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: JAX_ROUNDTRIP
// RUN: JAX_GENERIC_ROUNDTRIP
// RUN: XDSL_JAX_ROUNDTRIP
// RUN: XDSL_JAX_GENERIC_ROUNDTRIP

// CHECK: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
%t0 = "test.op"() : () -> tensor<i32>

// CHECK: %[[TRANSPOSE_OPERAND:.*]] = "test.op"() : () -> tensor<2x3x2xi32>
// CHECK-GENERIC: %[[TRANSPOSE_OPERAND:.*]] = "test.op"() : () -> tensor<2x3x2xi32>
%transpose_operand = "test.op"() : () -> tensor<2x3x2xi32>
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
// CHECK: %[[TRANSPOSE_RESULT:.*]] = stablehlo.transpose %[[TRANSPOSE_OPERAND]], dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// CHECK-GENERIC: %[[TRANSPOSE_RESULT:.*]] = "stablehlo.transpose"(%[[TRANSPOSE_OPERAND]]) <{permutation = array<i64: 2, 1, 0>}> : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
%transpose_result = stablehlo.transpose %transpose_operand, dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// %result: [
//           [[1,7], [3,9], [5,11]],
//           [[2,8], [4,10], [6,12]]
//          ]

// CHECK: %[[PAD_OPERAND:.*]] = "test.op"() : () -> tensor<2x3xi32>
// CHECK-GENERIC: %[[PAD_OPERAND:.*]] = "test.op"() : () -> tensor<2x3xi32>
%pad_operand = "test.op"() : () -> tensor<2x3xi32>
// CHECK: %[[PADDING_VALUE:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[PADDING_VALUE:.*]] = "test.op"() : () -> tensor<i32>
%padding_value = "test.op"() : () -> tensor<i32>
// %operand: [
//            [1, 2, 3],
//            [4, 5, 6]
//           ]
// %padding_value: 0
// CHECK: %[[PAD_RESULT:.*]] = stablehlo.pad %[[PAD_OPERAND]], %[[PADDING_VALUE]],
// CHECK:   low = [0, 1],
// CHECK:   high = [2, 1],
// CHECK:   interior = [1, 2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// CHECK-GENERIC: %[[PAD_RESULT:.*]] = "stablehlo.pad"(%[[PAD_OPERAND]], %[[PADDING_VALUE]])
// CHECK-GENERIC-DAG: edge_padding_low = array<i64: 0, 1>
// CHECK-GENERIC-DAG: edge_padding_high = array<i64: 2, 1>
// CHECK-GENERIC-DAG: interior_padding = array<i64: 1, 2>
// CHECK-GENERIC: : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
%pad_result = stablehlo.pad %pad_operand, %padding_value,
  low = [0, 1],
  high = [2, 1],
  interior = [1, 2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// %result: [
//           [0, 1, 0, 0, 2, 0, 0, 3, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 4, 0, 0, 5, 0, 0, 6, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0]
//          ]

// CHECK: %[[SLICE_INPUT:.*]] = "test.op"() : () -> tensor<3x8xi64>
%slice_input = "test.op"() : () -> tensor<3x8xi64>
// CHECK: %[[SLICE_RES:.*]] = stablehlo.slice %[[SLICE_INPUT]] [1:3, 4:8:2] : (tensor<3x8xi64>) -> tensor<2x2xi64>
// CHECK-GENERIC: %[[SLICE_INPUT_GEN:.*]] = "test.op"() : () -> tensor<3x8xi64>
// CHECK-GENERIC: %[[SLICE_RES_GEN:.*]] = "stablehlo.slice"(%[[SLICE_INPUT_GEN]]) <{
// CHECK-GENERIC-DAG: start_indices = array<i64: 1, 4>
// CHECK-GENERIC-DAG: limit_indices = array<i64: 3, 8>
// CHECK-GENERIC-DAG: strides = array<i64: 1, 2>
// CHECK-GENERIC: }> : (tensor<3x8xi64>) -> tensor<2x2xi64>
%slice = stablehlo.slice %slice_input [1:3, 4:8:2] : (tensor<3x8xi64>) -> tensor<2x2xi64>

// CHECK: %[[DYN_OPERAND:.*]] = "test.op"() : () -> tensor<4x4xi32>
// CHECK-GENERIC: %[[DYN_OPERAND:.*]] = "test.op"() : () -> tensor<4x4xi32>
%dyn_operand = "test.op"() : () -> tensor<4x4xi32>
// CHECK: %[[START0:.*]] = "test.op"() : () -> tensor<i64>
// CHECK-GENERIC: %[[START0:.*]] = "test.op"() : () -> tensor<i64>
%start0 = "test.op"() : () -> tensor<i64>
// CHECK: %[[START1:.*]] = "test.op"() : () -> tensor<i64>
// CHECK-GENERIC: %[[START1:.*]] = "test.op"() : () -> tensor<i64>
%start1 = "test.op"() : () -> tensor<i64>
// CHECK: %[[DYNAMIC_SLICE:.*]] = stablehlo.dynamic_slice %[[DYN_OPERAND]], %[[START0]], %[[START1]], sizes = [2, 3] : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x3xi32>
// CHECK-GENERIC: %[[DYNAMIC_SLICE:.*]] = "stablehlo.dynamic_slice"(%[[DYN_OPERAND]], %[[START0]], %[[START1]]) <{slice_sizes = array<i64: 2, 3>}> : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x3xi32>
%dynamic_slice = stablehlo.dynamic_slice %dyn_operand, %start0, %start1, sizes = [2, 3] : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x3xi32>

// CHECK: %[[RESHAPE_INPUT:.*]] = "test.op"() : () -> tensor<2xf32>
// CHECK-GENERIC: %[[RESHAPE_INPUT:.*]] = "test.op"() : () -> tensor<2xf32>
%reshape_input = "test.op"() : () -> tensor<2xf32>
// CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHAPE_INPUT]] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-GENERIC: %[[RESHAPE:.*]] = "stablehlo.reshape"(%[[RESHAPE_INPUT]]) : (tensor<2xf32>) -> tensor<1x2xf32>
%reshape = stablehlo.reshape %reshape_input : (tensor<2xf32>) -> tensor<1x2xf32>

// CHECK: %[[RESHAPE_DYNAMIC_INPUT:.*]] = "test.op"() : () -> tensor<?x3xi32>
// CHECK-GENERIC: %[[RESHAPE_DYNAMIC_INPUT:.*]] = "test.op"() : () -> tensor<?x3xi32>
%reshape_dynamic_input = "test.op"() : () -> tensor<?x3xi32>
// CHECK: %[[RESHAPE_DYNAMIC:.*]] = stablehlo.reshape %[[RESHAPE_DYNAMIC_INPUT]] : (tensor<?x3xi32>) -> tensor<6xi32>
// CHECK-GENERIC: %[[RESHAPE_DYNAMIC:.*]] = "stablehlo.reshape"(%[[RESHAPE_DYNAMIC_INPUT]]) : (tensor<?x3xi32>) -> tensor<6xi32>
%reshape_dynamic = stablehlo.reshape %reshape_dynamic_input : (tensor<?x3xi32>) -> tensor<6xi32>

// CHECK: %[[CONCAT_INPUT1:.*]] = "test.op"() : () -> tensor<3x2xi64>
// CHECK-GENERIC: %[[CONCAT_INPUT1:.*]] = "test.op"() : () -> tensor<3x2xi64>
%input1 = "test.op"() : () -> tensor<3x2xi64>
// CHECK: %[[CONCAT_INPUT2:.*]] = "test.op"() : () -> tensor<1x2xi64>
// CHECK-GENERIC: %[[CONCAT_INPUT2:.*]] = "test.op"() : () -> tensor<1x2xi64>
%input2 = "test.op"() : () -> tensor<1x2xi64>
// CHECK: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[CONCAT_INPUT1]], %[[CONCAT_INPUT2]], dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
// CHECK-GENERIC: %[[CONCATENATE:.*]] = "stablehlo.concatenate"(%[[CONCAT_INPUT1]], %[[CONCAT_INPUT2]]) <{dimension = 0 : i64}> : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
%concatenate = stablehlo.concatenate %input1, %input2, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>

// CHECK: %[[BROADCAST_INPUT:.*]] = "test.op"() : () -> tensor<1x3xi32>
// CHECK-GENERIC: %[[BROADCAST_INPUT:.*]] = "test.op"() : () -> tensor<1x3xi32>
%broadcast_input = "test.op"() : () -> tensor<1x3xi32>
// CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[BROADCAST_INPUT]], dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
// CHECK-GENERIC: %[[BROADCAST:.*]] = "stablehlo.broadcast_in_dim"(%[[BROADCAST_INPUT]]) <{broadcast_dimensions = array<i64: 2, 1>}> : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
%broadcast = stablehlo.broadcast_in_dim %broadcast_input, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>

// CHECK: %[[BROADCAST_DYNAMIC_INPUT:.*]] = "test.op"() : () -> tensor<?x3xi32>
// CHECK-GENERIC: %[[BROADCAST_DYNAMIC_INPUT:.*]] = "test.op"() : () -> tensor<?x3xi32>
%broadcast_dynamic_input = "test.op"() : () -> tensor<?x3xi32>
// CHECK: %[[BROADCAST_DYNAMIC:.*]] = stablehlo.broadcast_in_dim %[[BROADCAST_DYNAMIC_INPUT]], dims = [2, 1] : (tensor<?x3xi32>) -> tensor<2x3x2xi32>
// CHECK-GENERIC: %[[BROADCAST_DYNAMIC:.*]] = "stablehlo.broadcast_in_dim"(%[[BROADCAST_DYNAMIC_INPUT]]) <{broadcast_dimensions = array<i64: 2, 1>}> : (tensor<?x3xi32>) -> tensor<2x3x2xi32>
%broadcast_dynamic = stablehlo.broadcast_in_dim %broadcast_dynamic_input, dims = [2, 1] : (tensor<?x3xi32>) -> tensor<2x3x2xi32>

// CHECK: %[[GATHER_INPUT:.*]] = "test.op"() : () -> tensor<2x3x4x2xi32>
%gather_input = "test.op"() : () -> tensor<2x3x4x2xi32>
// CHECK: %[[START_INDICES:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
%start_indices = "test.op"() : () -> tensor<2x2x3x2xi64>
// CHECK: %[[GATHER_RES:.*]] = "stablehlo.gather"(%[[GATHER_INPUT]], %[[START_INDICES]])
// CHECK-DAG: dimension_numbers = #stablehlo.gather<offset_dims = [3, 4], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [1], start_index_map = [2, 1], index_vector_dim = 3>
// CHECK-DAG: slice_sizes = array<i64: 1, 1, 2, 2>
// CHECK-DAG: indices_are_sorted = false
// CHECK: : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
// CHECK-GENERIC: %[[GATHER_INPUT_GEN:.*]] = "test.op"() : () -> tensor<2x3x4x2xi32>
// CHECK-GENERIC: %[[START_INDICES_GEN:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
// CHECK-GENERIC: %[[GATHER_RES_GEN:.*]] = "stablehlo.gather"(%[[GATHER_INPUT_GEN]], %[[START_INDICES_GEN]]) <{
// CHECK-GENERIC-DAG: dimension_numbers = #stablehlo.gather<offset_dims = [3, 4], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [1], start_index_map = [2, 1], index_vector_dim = 3>
// CHECK-GENERIC-DAG: slice_sizes = array<i64: 1, 1, 2, 2>
// CHECK-GENERIC-DAG: indices_are_sorted = false
// CHECK-GENERIC: : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
%gather = "stablehlo.gather"(%gather_input, %start_indices) <{
  dimension_numbers = #stablehlo.gather<
    offset_dims = [3, 4],
    collapsed_slice_dims = [1],
    operand_batching_dims = [0],
    start_indices_batching_dims = [1],
    start_index_map = [2, 1],
    index_vector_dim = 3>,
  slice_sizes = array<i64: 1, 1, 2, 2>,
  indices_are_sorted = false
}> : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>

// CHECK: %[[SCATTER_INPUT:.*]] = "test.op"() : () -> tensor<2x3x4x2xi64>
%scatter_input = "test.op"() : () -> tensor<2x3x4x2xi64>
// CHECK: %[[SCATTER_INDICES:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
%scatter_indices = "test.op"() : () -> tensor<2x2x3x2xi64>
// CHECK: %[[SCATTER_UPDATES:.*]] = "test.op"() : () -> tensor<2x2x3x2x2xi64>
%scatter_updates = "test.op"() : () -> tensor<2x2x3x2x2xi64>
// CHECK: %[[SCATTER_RES:.*]] = "stablehlo.scatter"(%[[SCATTER_INPUT]], %[[SCATTER_INDICES]], %[[SCATTER_UPDATES]])
// CHECK-DAG: scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [3, 4], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [2, 1], index_vector_dim = 3>
// CHECK-DAG: indices_are_sorted = false
// CHECK-DAG: unique_indices = false
// CHECK: %[[SCATTER_ADD:.*]] = stablehlo.add %[[SCATTER_ARG0:[^,]*]], %[[SCATTER_ARG1:[^ ]*]] : tensor<i64>
// CHECK: stablehlo.return %[[SCATTER_ADD]] : tensor<i64>
// CHECK: -> tensor<2x3x4x2xi64>
// CHECK-GENERIC: %[[SCATTER_INPUT_GEN:.*]] = "test.op"() : () -> tensor<2x3x4x2xi64>
// CHECK-GENERIC: %[[SCATTER_INDICES_GEN:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
// CHECK-GENERIC: %[[SCATTER_UPDATES_GEN:.*]] = "test.op"() : () -> tensor<2x2x3x2x2xi64>
// CHECK-GENERIC: %[[SCATTER_RES_GEN:.*]] = "stablehlo.scatter"(%[[SCATTER_INPUT_GEN]], %[[SCATTER_INDICES_GEN]], %[[SCATTER_UPDATES_GEN]])
// CHECK-GENERIC-DAG: scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [3, 4], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [2, 1], index_vector_dim = 3>
// CHECK-GENERIC-DAG: indices_are_sorted = false
// CHECK-GENERIC-DAG: unique_indices = false
// CHECK-GENERIC: : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
%scatter = "stablehlo.scatter"(%scatter_input, %scatter_indices, %scatter_updates) <{
  scatter_dimension_numbers = #stablehlo.scatter<
    update_window_dims = [3, 4],
    inserted_window_dims = [1],
    input_batching_dims = [0],
    scatter_indices_batching_dims = [1],
    scatter_dims_to_operand_dims = [2, 1],
    index_vector_dim = 3>,
  indices_are_sorted = false,
  unique_indices = false
}> ({
  ^bb0(%arg_scatter_0: tensor<i64>, %arg_scatter_1: tensor<i64>):
    %scatter_add = "stablehlo.add"(%arg_scatter_0, %arg_scatter_1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%scatter_add) : (tensor<i64>) -> ()
}) : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
