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

%input = "test.op"() : () -> tensor<2x3xi32>
// CHECK: Operation does not verify: reshape output must have a static shape.
%reshape = stablehlo.reshape %input : (tensor<2x3xi32>) -> tensor<?x3xi32>

// -----

%input = "test.op"() : () -> tensor<2x3xi32>
// CHECK: Operation does not verify: number of output elements (5) doesn't match expected number of elements (6)
%reshape = stablehlo.reshape %input : (tensor<2x3xi32>) -> tensor<5xi32>

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

%slice_input = "test.op"() : () -> tensor<3x8xi64>
// CHECK: Operation does not verify: all of {start_indices, limit_indices, strides} must have the same size
%slice = "stablehlo.slice"(%slice_input) <{start_indices = array<i64: 1, 4>, limit_indices = array<i64: 3>, strides = array<i64: 1, 2>}> : (tensor<3x8xi64>) -> tensor<2x2xi64>

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
