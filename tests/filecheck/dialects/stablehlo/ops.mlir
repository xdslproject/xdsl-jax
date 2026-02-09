// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%t0 = "test.op"() : () -> tensor<i32>
%tf32 = "test.op"() : () -> tensor<f32>

// Elementwise unary operations

// CHECK: %cbrt = stablehlo.cbrt %tf32 : tensor<f32>
// CHECK-GENERIC: %cbrt = "stablehlo.cbrt"(%tf32) : (tensor<f32>) -> tensor<f32>
%cbrt = stablehlo.cbrt %tf32 : tensor<f32>

// CHECK: %ceil = stablehlo.ceil %tf32 : tensor<f32>
// CHECK-GENERIC: %ceil = "stablehlo.ceil"(%tf32) : (tensor<f32>) -> tensor<f32>
%ceil = stablehlo.ceil %tf32 : tensor<f32>

// CHECK: %convert = stablehlo.convert %t0 : (tensor<i32>) -> tensor<f32>
// CHECK-GENERIC: %convert = "stablehlo.convert"(%t0) : (tensor<i32>) -> tensor<f32>
%convert = stablehlo.convert %t0 : (tensor<i32>) -> tensor<f32>

// CHECK: %count_leading_zeros = stablehlo.count_leading_zeros %t0 : tensor<i32>
// CHECK-GENERIC: %count_leading_zeros = "stablehlo.count_leading_zeros"(%t0) : (tensor<i32>) -> tensor<i32>
%count_leading_zeros = stablehlo.count_leading_zeros %t0 : tensor<i32>

// CHECK: %exponential_minus_one = stablehlo.exponential_minus_one %tf32 : tensor<f32>
// CHECK-GENERIC: %exponential_minus_one = "stablehlo.exponential_minus_one"(%tf32) : (tensor<f32>) -> tensor<f32>
%exponential_minus_one = stablehlo.exponential_minus_one %tf32 : tensor<f32>

// CHECK: %exponential = stablehlo.exponential %tf32 : tensor<f32>
// CHECK-GENERIC: %exponential = "stablehlo.exponential"(%tf32) : (tensor<f32>) -> tensor<f32>
%exponential = stablehlo.exponential %tf32 : tensor<f32>

// CHECK: %is_finite = stablehlo.is_finite %tf32 : (tensor<f32>) -> tensor<i1>
// CHECK-GENERIC: %is_finite = "stablehlo.is_finite"(%tf32) : (tensor<f32>) -> tensor<i1>
%is_finite = stablehlo.is_finite %tf32 : (tensor<f32>) -> tensor<i1>

// CHECK: %floor = stablehlo.floor %tf32 : tensor<f32>
// CHECK-GENERIC: %floor = "stablehlo.floor"(%tf32) : (tensor<f32>) -> tensor<f32>
%floor = stablehlo.floor %tf32 : tensor<f32>

// CHECK: %logistic = stablehlo.logistic %tf32 : tensor<f32>
// CHECK-GENERIC: %logistic = "stablehlo.logistic"(%tf32) : (tensor<f32>) -> tensor<f32>
%logistic = stablehlo.logistic %tf32 : tensor<f32>

// CHECK: %log = stablehlo.log %tf32 : tensor<f32>
// CHECK-GENERIC: %log = "stablehlo.log"(%tf32) : (tensor<f32>) -> tensor<f32>
%log = stablehlo.log %tf32 : tensor<f32>

// CHECK: %log_plus_one = stablehlo.log_plus_one %tf32 : tensor<f32>
// CHECK-GENERIC: %log_plus_one = "stablehlo.log_plus_one"(%tf32) : (tensor<f32>) -> tensor<f32>
%log_plus_one = stablehlo.log_plus_one %tf32 : tensor<f32>

// CHECK: %not = stablehlo.not %t0 : tensor<i32>
// CHECK-GENERIC: %not = "stablehlo.not"(%t0) : (tensor<i32>) -> tensor<i32>
%not = stablehlo.not %t0 : tensor<i32>

// CHECK: %popcnt = stablehlo.popcnt %t0 : tensor<i32>
// CHECK-GENERIC: %popcnt = "stablehlo.popcnt"(%t0) : (tensor<i32>) -> tensor<i32>
%popcnt = stablehlo.popcnt %t0 : tensor<i32>

// CHECK: %round_nearest_afz = stablehlo.round_nearest_afz %tf32 : tensor<f32>
// CHECK-GENERIC: %round_nearest_afz = "stablehlo.round_nearest_afz"(%tf32) : (tensor<f32>) -> tensor<f32>
%round_nearest_afz = stablehlo.round_nearest_afz %tf32 : tensor<f32>

// CHECK: %round_nearest_even = stablehlo.round_nearest_even %tf32 : tensor<f32>
// CHECK-GENERIC: %round_nearest_even = "stablehlo.round_nearest_even"(%tf32) : (tensor<f32>) -> tensor<f32>
%round_nearest_even = stablehlo.round_nearest_even %tf32 : tensor<f32>

// CHECK: %rsqrt = stablehlo.rsqrt %tf32 : tensor<f32>
// CHECK-GENERIC: %rsqrt = "stablehlo.rsqrt"(%tf32) : (tensor<f32>) -> tensor<f32>
%rsqrt = stablehlo.rsqrt %tf32 : tensor<f32>

// CHECK: %sqrt = stablehlo.sqrt %tf32 : tensor<f32>
// CHECK-GENERIC: %sqrt = "stablehlo.sqrt"(%tf32) : (tensor<f32>) -> tensor<f32>
%sqrt = stablehlo.sqrt %tf32 : tensor<f32>

// Other operations

// CHECK-GENERIC: %abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>
%abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>

// CHECK-GENERIC: %add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

%token0 = "test.op"() : () -> !stablehlo.token
%token1 = "test.op"() : () -> !stablehlo.token
// CHECK-GENERIC: %after_all = "stablehlo.after_all"(%token0, %token1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
%after_all = "stablehlo.after_all"(%token0, %token1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token

// CHECK-GENERIC: %atan2 = "stablehlo.atan2"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%atan2 = "stablehlo.atan2"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>

// CHECK-GENERIC: %multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK-GENERIC: %subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

%transpose_operand = "test.op"() : () -> tensor<2x3x2xi32>
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
// CHECK-GENERIC:  %transpose_result = "stablehlo.transpose"(%transpose_operand) {permutation = array<i64: 2, 1, 0>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
%transpose_result = "stablehlo.transpose"(%transpose_operand) {
  permutation = array<i64: 2, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// %result: [
//           [[1,7], [3,9], [5,11]],
//           [[2,8], [4,10], [6,12]]
//          ]

%pad_operand = "test.op"() : () -> tensor<2x3xi32>
%padding_value = "test.op"() : () -> tensor<i32>
// %operand: [
//            [1, 2, 3],
//            [4, 5, 6]
//           ]
// %padding_value: 0
%pad_result = "stablehlo.pad"(%pad_operand, %padding_value) {
  edge_padding_low = array<i64: 0, 1>,
  edge_padding_high = array<i64: 2, 1>,
  interior_padding = array<i64: 1, 2>
} : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// CHECK-GENERIC: %pad_result = "stablehlo.pad"(%pad_operand, %padding_value) {edge_padding_low = array<i64: 0, 1>, edge_padding_high = array<i64: 2, 1>, interior_padding = array<i64: 1, 2>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// %result: [
//           [0, 1, 0, 0, 2, 0, 0, 3, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 4, 0, 0, 5, 0, 0, 6, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0]
//          ]

// CHECK-GENERIC: %and = "stablehlo.and"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%and = "stablehlo.and"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK-GENERIC: %or = "stablehlo.or"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%or = "stablehlo.or"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK-GENERIC: %xor = "stablehlo.xor"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%xor = "stablehlo.xor"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK-GENERIC: %shift_left = "stablehlo.shift_left"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_left = "stablehlo.shift_left"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK-GENERIC: %shift_right_arithmetic = "stablehlo.shift_right_arithmetic"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_arithmetic = "stablehlo.shift_right_arithmetic"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK-GENERIC: %shift_right_logical = "stablehlo.shift_right_logical"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_logical = "stablehlo.shift_right_logical"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

// %bitcast = "stablehlo.bitcast_convert"(%t0) : (tensor<i32>) -> tensor<2xi16>
%bitcast = "stablehlo.bitcast_convert"(%t0) : (tensor<i32>) -> tensor<2xi16>

%index = "test.op"() : () -> tensor<i32>
%result_branch0 = "test.op"() : () -> tensor<2xi64>
%result_branch1 = "test.op"() : () -> tensor<2xi64>

// CHECK-GENERIC: %0, %1 = "stablehlo.case"(%index) ({
%0:2 = "stablehlo.case"(%index) ({
  // CHECK-GENERIC: "stablehlo.return"(%result_branch0, %result_branch0) : (tensor<2xi64>, tensor<2xi64>) -> ()
  "stablehlo.return"(%result_branch0, %result_branch0) : (tensor<2xi64>, tensor<2xi64>) -> ()
}, {
  // CHECK-GENERIC: "stablehlo.return"(%result_branch1, %result_branch1) : (tensor<2xi64>, tensor<2xi64>) -> ()
  "stablehlo.return"(%result_branch1, %result_branch1) : (tensor<2xi64>, tensor<2xi64>) -> ()
// CHECK-GENERIC: }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
}) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)

// CHECK-GENERIC: %constant = "stablehlo.constant"() {value = dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
%constant = "stablehlo.constant"() {value = dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
