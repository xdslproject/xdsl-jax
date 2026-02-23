// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%t0 = "test.op"() : () -> tensor<i32>
%tf32 = "test.op"() : () -> tensor<f32>
%t5f32 = "test.op"() : () -> tensor<5xf32>
%tcomplex = "test.op"() : () -> tensor<complex<f32>>


// Elementwise unary operations


// CHECK: %abs = stablehlo.abs %t0 : tensor<i32>
// CHECK-GENERIC: %abs = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>
%abs = stablehlo.abs %t0 : tensor<i32>

// CHECK: %complex_abs = stablehlo.abs %tcomplex : (tensor<complex<f32>>) -> tensor<f32>
// CHECK-GENERIC: %complex_abs = "stablehlo.abs"(%tcomplex) : (tensor<complex<f32>>) -> tensor<f32>
%complex_abs = stablehlo.abs %tcomplex : (tensor<complex<f32>>) -> tensor<f32>

// CHECK: %cbrt = stablehlo.cbrt %tf32 : tensor<f32>
// CHECK-GENERIC: %cbrt = "stablehlo.cbrt"(%tf32) : (tensor<f32>) -> tensor<f32>
%cbrt = stablehlo.cbrt %tf32 : tensor<f32>

// CHECK: %ceil = stablehlo.ceil %tf32 : tensor<f32>
// CHECK-GENERIC: %ceil = "stablehlo.ceil"(%tf32) : (tensor<f32>) -> tensor<f32>
%ceil = stablehlo.ceil %tf32 : tensor<f32>

// CHECK: %convert = stablehlo.convert %t0 : (tensor<i32>) -> tensor<f32>
// CHECK-GENERIC: %convert = "stablehlo.convert"(%t0) : (tensor<i32>) -> tensor<f32>
%convert = stablehlo.convert %t0 : (tensor<i32>) -> tensor<f32>

// CHECK: %cosine = stablehlo.cosine %tf32 : tensor<f32>
// CHECK-GENERIC: %cosine = "stablehlo.cosine"(%tf32) : (tensor<f32>) -> tensor<f32>
%cosine = stablehlo.cosine %tf32 : tensor<f32>

// CHECK: %count_leading_zeros = stablehlo.count_leading_zeros %t0 : tensor<i32>
// CHECK-GENERIC: %count_leading_zeros = "stablehlo.count_leading_zeros"(%t0) : (tensor<i32>) -> tensor<i32>
%count_leading_zeros = stablehlo.count_leading_zeros %t0 : tensor<i32>

// CHECK: %exponential_minus_one = stablehlo.exponential_minus_one %tf32 : tensor<f32>
// CHECK-GENERIC: %exponential_minus_one = "stablehlo.exponential_minus_one"(%tf32) : (tensor<f32>) -> tensor<f32>
%exponential_minus_one = stablehlo.exponential_minus_one %tf32 : tensor<f32>

// CHECK: %exponential = stablehlo.exponential %tf32 : tensor<f32>
// CHECK-GENERIC: %exponential = "stablehlo.exponential"(%tf32) : (tensor<f32>) -> tensor<f32>
%exponential = stablehlo.exponential %tf32 : tensor<f32>

// CHECK: %imag = stablehlo.imag %tcomplex : (tensor<complex<f32>>) -> tensor<f32>
// CHECK-GENERIC: %imag = "stablehlo.imag"(%tcomplex) : (tensor<complex<f32>>) -> tensor<f32>
%imag = stablehlo.imag %tcomplex : (tensor<complex<f32>>) -> tensor<f32>

// CHECK: %negate = stablehlo.negate %t0 : tensor<i32>
// CHECK-GENERIC: %negate = "stablehlo.negate"(%t0) : (tensor<i32>) -> tensor<i32>
%negate = stablehlo.negate %t0 : tensor<i32>

// CHECK: %floor = stablehlo.floor %tf32 : tensor<f32>
// CHECK-GENERIC: %floor = "stablehlo.floor"(%tf32) : (tensor<f32>) -> tensor<f32>
%floor = stablehlo.floor %tf32 : tensor<f32>

// CHECK: %is_finite = stablehlo.is_finite %tf32 : (tensor<f32>) -> tensor<i1>
// CHECK-GENERIC: %is_finite = "stablehlo.is_finite"(%tf32) : (tensor<f32>) -> tensor<i1>
%is_finite = stablehlo.is_finite %tf32 : (tensor<f32>) -> tensor<i1>

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

// CHECK: %real = stablehlo.real %tcomplex : (tensor<complex<f32>>) -> tensor<f32>
// CHECK-GENERIC: %real = "stablehlo.real"(%tcomplex) : (tensor<complex<f32>>) -> tensor<f32>
%real = stablehlo.real %tcomplex : (tensor<complex<f32>>) -> tensor<f32>

// CHECK: %round_nearest_afz = stablehlo.round_nearest_afz %tf32 : tensor<f32>
// CHECK-GENERIC: %round_nearest_afz = "stablehlo.round_nearest_afz"(%tf32) : (tensor<f32>) -> tensor<f32>
%round_nearest_afz = stablehlo.round_nearest_afz %tf32 : tensor<f32>

// CHECK: %round_nearest_even = stablehlo.round_nearest_even %tf32 : tensor<f32>
// CHECK-GENERIC: %round_nearest_even = "stablehlo.round_nearest_even"(%tf32) : (tensor<f32>) -> tensor<f32>
%round_nearest_even = stablehlo.round_nearest_even %tf32 : tensor<f32>

// CHECK: %rsqrt = stablehlo.rsqrt %tf32 : tensor<f32>
// CHECK-GENERIC: %rsqrt = "stablehlo.rsqrt"(%tf32) : (tensor<f32>) -> tensor<f32>
%rsqrt = stablehlo.rsqrt %tf32 : tensor<f32>

// CHECK: %sign = stablehlo.sign %tf32 : tensor<f32>
// CHECK-GENERIC: %sign = "stablehlo.sign"(%tf32) : (tensor<f32>) -> tensor<f32>
%sign = stablehlo.sign %tf32 : tensor<f32>

// CHECK: %sine = stablehlo.sine %tf32 : tensor<f32>
// CHECK-GENERIC: %sine = "stablehlo.sine"(%tf32) : (tensor<f32>) -> tensor<f32>
%sine = stablehlo.sine %tf32 : tensor<f32>

// CHECK: %sqrt = stablehlo.sqrt %tf32 : tensor<f32>
// CHECK-GENERIC: %sqrt = "stablehlo.sqrt"(%tf32) : (tensor<f32>) -> tensor<f32>
%sqrt = stablehlo.sqrt %tf32 : tensor<f32>

// CHECK: %tan = stablehlo.tan %tf32 : tensor<f32>
// CHECK-GENERIC: %tan = "stablehlo.tan"(%tf32) : (tensor<f32>) -> tensor<f32>
%tan = stablehlo.tan %tf32 : tensor<f32>

// CHECK: %tanh = stablehlo.tanh %tf32 : tensor<f32>
// CHECK-GENERIC: %tanh = "stablehlo.tanh"(%tf32) : (tensor<f32>) -> tensor<f32>
%tanh = stablehlo.tanh %tf32 : tensor<f32>


// Elementwise binary operations


// CHECK: %add = stablehlo.add %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %add = "stablehlo.add"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%add = stablehlo.add %t0, %t0 : tensor<i32>

// CHECK: %and = stablehlo.and %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %and = "stablehlo.and"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%and = stablehlo.and %t0, %t0 : tensor<i32>

// CHECK: %atan2 = stablehlo.atan2 %tf32, %tf32 : tensor<f32>
// CHECK-GENERIC: %atan2 = "stablehlo.atan2"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%atan2 = stablehlo.atan2 %tf32, %tf32 : tensor<f32>

// CHECK: %complex = stablehlo.complex %tf32, %tf32 : tensor<complex<f32>>
// CHECK-GENERIC: %complex = "stablehlo.complex"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
%complex = stablehlo.complex %tf32, %tf32 : tensor<complex<f32>>

// CHECK: %complex2 = stablehlo.complex %t5f32, %t5f32 : tensor<5xcomplex<f32>>
// CHECK-GENERIC: %complex2 = "stablehlo.complex"(%t5f32, %t5f32) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xcomplex<f32>>
%complex2 = stablehlo.complex %t5f32, %t5f32 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xcomplex<f32>>

// CHECK: %divide = stablehlo.divide %tf32, %tf32 : tensor<f32>
// CHECK-GENERIC: %divide = "stablehlo.divide"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%divide = stablehlo.divide %tf32, %tf32 : tensor<f32>

// CHECK: %maximum = stablehlo.maximum %tf32, %tf32 : tensor<f32>
// CHECK-GENERIC: %maximum = "stablehlo.maximum"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%maximum = stablehlo.maximum %tf32, %tf32 : tensor<f32>

// CHECK: %minimum = stablehlo.minimum %tf32, %tf32 : tensor<f32>
// CHECK-GENERIC: %minimum = "stablehlo.minimum"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%minimum = stablehlo.minimum %tf32, %tf32 : tensor<f32>

// CHECK: %multiply = stablehlo.multiply %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %multiply = "stablehlo.multiply"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%multiply = stablehlo.multiply %t0, %t0 : tensor<i32>

// CHECK: %or = stablehlo.or %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %or = "stablehlo.or"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%or = stablehlo.or %t0, %t0 : tensor<i32>

// CHECK: %power = stablehlo.power %tf32, %tf32 : tensor<f32>
// CHECK-GENERIC: %power = "stablehlo.power"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%power = stablehlo.power %tf32, %tf32 : tensor<f32>

// CHECK: %remainder = stablehlo.remainder %tf32, %tf32 : tensor<f32>
// CHECK-GENERIC: %remainder = "stablehlo.remainder"(%tf32, %tf32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%remainder = stablehlo.remainder %tf32, %tf32 : tensor<f32>

// CHECK: %shift_left = stablehlo.shift_left %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %shift_left = "stablehlo.shift_left"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_left = stablehlo.shift_left %t0, %t0 : tensor<i32>

// CHECK: %shift_right_arithmetic = stablehlo.shift_right_arithmetic %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %shift_right_arithmetic = "stablehlo.shift_right_arithmetic"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_arithmetic = stablehlo.shift_right_arithmetic %t0, %t0 : tensor<i32>

// CHECK: %shift_right_logical = stablehlo.shift_right_logical %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %shift_right_logical = "stablehlo.shift_right_logical"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_logical = stablehlo.shift_right_logical %t0, %t0 : tensor<i32>

// CHECK: %subtract = stablehlo.subtract %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %subtract = "stablehlo.subtract"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%subtract = stablehlo.subtract %t0, %t0 : tensor<i32>

// CHECK: %xor = stablehlo.xor %t0, %t0 : tensor<i32>
// CHECK-GENERIC: %xor = "stablehlo.xor"(%t0, %t0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%xor = stablehlo.xor %t0, %t0 : tensor<i32>

// Other operations

%token0 = "test.op"() : () -> !stablehlo.token
%token1 = "test.op"() : () -> !stablehlo.token
// CHECK: %after_all = stablehlo.after_all %token0, %token1 : !stablehlo.token
// CHECK-GENERIC: %after_all = "stablehlo.after_all"(%token0, %token1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
%after_all = stablehlo.after_all %token0, %token1 : !stablehlo.token

%transpose_operand = "test.op"() : () -> tensor<2x3x2xi32>
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
// CHECK: %transpose_result = stablehlo.transpose %transpose_operand, dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// CHECK-GENERIC: %transpose_result = "stablehlo.transpose"(%transpose_operand) {permutation = array<i64: 2, 1, 0>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
%transpose_result = stablehlo.transpose %transpose_operand, dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
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
// CHECK: %pad_result = stablehlo.pad %pad_operand, %padding_value, 
//   low = [0, 1], 
//   high = [2, 1], 
//   interior = [1, 2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// CHECK-GENERIC: %pad_result = "stablehlo.pad"(%pad_operand, %padding_value) {
//   edge_padding_low = array<i64: 0, 1>,
//   edge_padding_high = array<i64: 2, 1>,
//   interior_padding = array<i64: 1, 2>
// } : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
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

// CHECK: %bitcast = stablehlo.bitcast_convert %t0 : (tensor<i32>) -> tensor<2xi16>
// CHECK-GENERIC: %bitcast = "stablehlo.bitcast_convert"(%t0) : (tensor<i32>) -> tensor<2xi16>
%bitcast = stablehlo.bitcast_convert %t0 : (tensor<i32>) -> tensor<2xi16>

%index = "test.op"() : () -> tensor<i32>
%result_branch0 = "test.op"() : () -> tensor<2xi64>
%result_branch1 = "test.op"() : () -> tensor<2xi64>

// CHECK-GENERIC: %0, %1 = "stablehlo.case"(%index) ({
%0:2 = "stablehlo.case"(%index) ({
  // CHECK: stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  // CHECK-GENERIC: "stablehlo.return"(%result_branch0, %result_branch0) : (tensor<2xi64>, tensor<2xi64>) -> ()
  stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
}, {
  // CHECK: stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  // CHECK-GENERIC: "stablehlo.return"(%result_branch1, %result_branch1) : (tensor<2xi64>, tensor<2xi64>) -> ()
  stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
// CHECK-GENERIC: }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
}) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)

// CHECK: %constant = stablehlo.constant dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>
// CHECK-GENERIC: %constant = "stablehlo.constant"() {value = dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
%constant = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
