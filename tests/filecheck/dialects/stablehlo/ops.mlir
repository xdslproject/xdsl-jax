// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: %[[PRED:.*]] = "test.op"() : () -> tensor<i1>
// CHECK-GENERIC: %[[PRED:.*]] = "test.op"() : () -> tensor<i1>
%pred = "test.op"() : () -> tensor<i1>
// CHECK: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
%t0 = "test.op"() : () -> tensor<i32>
// CHECK: %[[TF32:.*]] = "test.op"() : () -> tensor<f32>
// CHECK-GENERIC: %[[TF32:.*]] = "test.op"() : () -> tensor<f32>
%tf32 = "test.op"() : () -> tensor<f32>
// CHECK: %[[TF64:.*]] = "test.op"() : () -> tensor<f64>
// CHECK-GENERIC: %[[TF64:.*]] = "test.op"() : () -> tensor<f64>
%tf64 = "test.op"() : () -> tensor<f64>
// CHECK: %[[T5F32:.*]] = "test.op"() : () -> tensor<5xf32>
// CHECK-GENERIC: %[[T5F32:.*]] = "test.op"() : () -> tensor<5xf32>
%t5f32 = "test.op"() : () -> tensor<5xf32>
// CHECK: %[[TCOMPLEX:.*]] = "test.op"() : () -> tensor<complex<f32>>
// CHECK-GENERIC: %[[TCOMPLEX:.*]] = "test.op"() : () -> tensor<complex<f32>>
%tcomplex = "test.op"() : () -> tensor<complex<f32>>


// Elementwise unary operations


// CHECK: %[[ABS:.*]] = stablehlo.abs %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[ABS:.*]] = "stablehlo.abs"(%[[T0]]) : (tensor<i32>) -> tensor<i32>
%abs = stablehlo.abs %t0 : tensor<i32>

// CHECK: %[[COMPLEX_ABS:.*]] = stablehlo.abs %[[TCOMPLEX]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK-GENERIC: %[[COMPLEX_ABS:.*]] = "stablehlo.abs"(%[[TCOMPLEX]]) : (tensor<complex<f32>>) -> tensor<f32>
%complex_abs = stablehlo.abs %tcomplex : (tensor<complex<f32>>) -> tensor<f32>

// CHECK: %[[CBRT:.*]] = stablehlo.cbrt %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[CBRT:.*]] = "stablehlo.cbrt"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%cbrt = stablehlo.cbrt %tf32 : tensor<f32>

// CHECK: %[[CEIL:.*]] = stablehlo.ceil %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[CEIL:.*]] = "stablehlo.ceil"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%ceil = stablehlo.ceil %tf32 : tensor<f32>

// CHECK: %[[CONVERT:.*]] = stablehlo.convert %[[T0]] : (tensor<i32>) -> tensor<f32>
// CHECK-GENERIC: %[[CONVERT:.*]] = "stablehlo.convert"(%[[T0]]) : (tensor<i32>) -> tensor<f32>
%convert = stablehlo.convert %t0 : (tensor<i32>) -> tensor<f32>

// CHECK: %[[COSINE:.*]] = stablehlo.cosine %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[COSINE:.*]] = "stablehlo.cosine"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%cosine = stablehlo.cosine %tf32 : tensor<f32>

// CHECK: %[[COUNT_LEADING_ZEROS:.*]] = stablehlo.count_leading_zeros %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[COUNT_LEADING_ZEROS:.*]] = "stablehlo.count_leading_zeros"(%[[T0]]) : (tensor<i32>) -> tensor<i32>
%count_leading_zeros = stablehlo.count_leading_zeros %t0 : tensor<i32>

// CHECK: %[[EXPONENTIAL_MINUS_ONE:.*]] = stablehlo.exponential_minus_one %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[EXPONENTIAL_MINUS_ONE:.*]] = "stablehlo.exponential_minus_one"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%exponential_minus_one = stablehlo.exponential_minus_one %tf32 : tensor<f32>

// CHECK: %[[EXPONENTIAL:.*]] = stablehlo.exponential %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[EXPONENTIAL:.*]] = "stablehlo.exponential"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%exponential = stablehlo.exponential %tf32 : tensor<f32>

// CHECK: %[[IMAG:.*]] = stablehlo.imag %[[TCOMPLEX]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK-GENERIC: %[[IMAG:.*]] = "stablehlo.imag"(%[[TCOMPLEX]]) : (tensor<complex<f32>>) -> tensor<f32>
%imag = stablehlo.imag %tcomplex : (tensor<complex<f32>>) -> tensor<f32>

// CHECK: %[[NEGATE:.*]] = stablehlo.negate %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[NEGATE:.*]] = "stablehlo.negate"(%[[T0]]) : (tensor<i32>) -> tensor<i32>
%negate = stablehlo.negate %t0 : tensor<i32>

// CHECK: %[[FLOOR:.*]] = stablehlo.floor %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[FLOOR:.*]] = "stablehlo.floor"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%floor = stablehlo.floor %tf32 : tensor<f32>

// CHECK: %[[IS_FINITE:.*]] = stablehlo.is_finite %[[TF32]] : (tensor<f32>) -> tensor<i1>
// CHECK-GENERIC: %[[IS_FINITE:.*]] = "stablehlo.is_finite"(%[[TF32]]) : (tensor<f32>) -> tensor<i1>
%is_finite = stablehlo.is_finite %tf32 : (tensor<f32>) -> tensor<i1>

// CHECK: %[[LOGISTIC:.*]] = stablehlo.logistic %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[LOGISTIC:.*]] = "stablehlo.logistic"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%logistic = stablehlo.logistic %tf32 : tensor<f32>

// CHECK: %[[LOG:.*]] = stablehlo.log %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[LOG:.*]] = "stablehlo.log"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%log = stablehlo.log %tf32 : tensor<f32>

// CHECK: %[[LOG_PLUS_ONE:.*]] = stablehlo.log_plus_one %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[LOG_PLUS_ONE:.*]] = "stablehlo.log_plus_one"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%log_plus_one = stablehlo.log_plus_one %tf32 : tensor<f32>

// CHECK: %[[NOT:.*]] = stablehlo.not %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[NOT:.*]] = "stablehlo.not"(%[[T0]]) : (tensor<i32>) -> tensor<i32>
%not = stablehlo.not %t0 : tensor<i32>

// CHECK: %[[POPCNT:.*]] = stablehlo.popcnt %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[POPCNT:.*]] = "stablehlo.popcnt"(%[[T0]]) : (tensor<i32>) -> tensor<i32>
%popcnt = stablehlo.popcnt %t0 : tensor<i32>

// CHECK: %[[REAL:.*]] = stablehlo.real %[[TCOMPLEX]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK-GENERIC: %[[REAL:.*]] = "stablehlo.real"(%[[TCOMPLEX]]) : (tensor<complex<f32>>) -> tensor<f32>
%real = stablehlo.real %tcomplex : (tensor<complex<f32>>) -> tensor<f32>

// CHECK: %[[ROUND_NEAREST_AFZ:.*]] = stablehlo.round_nearest_afz %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[ROUND_NEAREST_AFZ:.*]] = "stablehlo.round_nearest_afz"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%round_nearest_afz = stablehlo.round_nearest_afz %tf32 : tensor<f32>

// CHECK: %[[ROUND_NEAREST_EVEN:.*]] = stablehlo.round_nearest_even %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[ROUND_NEAREST_EVEN:.*]] = "stablehlo.round_nearest_even"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%round_nearest_even = stablehlo.round_nearest_even %tf32 : tensor<f32>

// CHECK: %[[RSQRT:.*]] = stablehlo.rsqrt %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[RSQRT:.*]] = "stablehlo.rsqrt"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%rsqrt = stablehlo.rsqrt %tf32 : tensor<f32>

// CHECK: %[[SIGN:.*]] = stablehlo.sign %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[SIGN:.*]] = "stablehlo.sign"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%sign = stablehlo.sign %tf32 : tensor<f32>

// CHECK: %[[SINE:.*]] = stablehlo.sine %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[SINE:.*]] = "stablehlo.sine"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%sine = stablehlo.sine %tf32 : tensor<f32>

// CHECK: %[[SQRT:.*]] = stablehlo.sqrt %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[SQRT:.*]] = "stablehlo.sqrt"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%sqrt = stablehlo.sqrt %tf32 : tensor<f32>

// CHECK: %[[TAN:.*]] = stablehlo.tan %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[TAN:.*]] = "stablehlo.tan"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%tan = stablehlo.tan %tf32 : tensor<f32>

// CHECK: %[[TANH:.*]] = stablehlo.tanh %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[TANH:.*]] = "stablehlo.tanh"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%tanh = stablehlo.tanh %tf32 : tensor<f32>


// Elementwise binary operations


// CHECK: %[[ADD:.*]] = stablehlo.add %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[ADD:.*]] = "stablehlo.add"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%add = stablehlo.add %t0, %t0 : tensor<i32>

// CHECK: %[[AND:.*]] = stablehlo.and %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[AND:.*]] = "stablehlo.and"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%and = stablehlo.and %t0, %t0 : tensor<i32>

// CHECK: %[[ATAN2:.*]] = stablehlo.atan2 %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[ATAN2:.*]] = "stablehlo.atan2"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%atan2 = stablehlo.atan2 %tf32, %tf32 : tensor<f32>

// CHECK: %[[COMPLEX:.*]] = stablehlo.complex %[[TF32]], %[[TF32]] : tensor<complex<f32>>
// CHECK-GENERIC: %[[COMPLEX:.*]] = "stablehlo.complex"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
%complex = stablehlo.complex %tf32, %tf32 : tensor<complex<f32>>

// CHECK: %[[COMPLEX2:.*]] = stablehlo.complex %[[T5F32]], %[[T5F32]] : tensor<5xcomplex<f32>>
// CHECK-GENERIC: %[[COMPLEX2:.*]] = "stablehlo.complex"(%[[T5F32]], %[[T5F32]]) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xcomplex<f32>>
%complex2 = stablehlo.complex %t5f32, %t5f32 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xcomplex<f32>>

// CHECK: %[[COMPLEX_FALLBACK:.*]] = stablehlo.complex %[[TF64]], %[[TF64]] : tensor<complex<f64>>
// CHECK-GENERIC: %[[COMPLEX_FALLBACK:.*]] = "stablehlo.complex"(%[[TF64]], %[[TF64]]) : (tensor<f64>, tensor<f64>) -> tensor<complex<f64>>
%complex_fallback = stablehlo.complex %tf64, %tf64 : (tensor<f64>, tensor<f64>) -> tensor<complex<f64>>

// CHECK: %[[DIVIDE:.*]] = stablehlo.divide %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[DIVIDE:.*]] = "stablehlo.divide"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%divide = stablehlo.divide %tf32, %tf32 : tensor<f32>

// CHECK: %[[MAXIMUM:.*]] = stablehlo.maximum %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[MAXIMUM:.*]] = "stablehlo.maximum"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%maximum = stablehlo.maximum %tf32, %tf32 : tensor<f32>

// CHECK: %[[MINIMUM:.*]] = stablehlo.minimum %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[MINIMUM:.*]] = "stablehlo.minimum"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%minimum = stablehlo.minimum %tf32, %tf32 : tensor<f32>

// CHECK: %[[MULTIPLY:.*]] = stablehlo.multiply %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[MULTIPLY:.*]] = "stablehlo.multiply"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%multiply = stablehlo.multiply %t0, %t0 : tensor<i32>

// CHECK: %[[OR:.*]] = stablehlo.or %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[OR:.*]] = "stablehlo.or"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%or = stablehlo.or %t0, %t0 : tensor<i32>

// CHECK: %[[POWER:.*]] = stablehlo.power %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[POWER:.*]] = "stablehlo.power"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%power = stablehlo.power %tf32, %tf32 : tensor<f32>

// CHECK: %[[REMAINDER:.*]] = stablehlo.remainder %[[TF32]], %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[REMAINDER:.*]] = "stablehlo.remainder"(%[[TF32]], %[[TF32]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
%remainder = stablehlo.remainder %tf32, %tf32 : tensor<f32>

// CHECK: %[[SHIFT_LEFT:.*]] = stablehlo.shift_left %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SHIFT_LEFT:.*]] = "stablehlo.shift_left"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_left = stablehlo.shift_left %t0, %t0 : tensor<i32>

// CHECK: %[[SHIFT_RIGHT_ARITHMETIC:.*]] = stablehlo.shift_right_arithmetic %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SHIFT_RIGHT_ARITHMETIC:.*]] = "stablehlo.shift_right_arithmetic"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_arithmetic = stablehlo.shift_right_arithmetic %t0, %t0 : tensor<i32>

// CHECK: %[[SHIFT_RIGHT_LOGICAL:.*]] = stablehlo.shift_right_logical %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SHIFT_RIGHT_LOGICAL:.*]] = "stablehlo.shift_right_logical"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%shift_right_logical = stablehlo.shift_right_logical %t0, %t0 : tensor<i32>

// CHECK: %[[SUBTRACT:.*]] = stablehlo.subtract %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[SUBTRACT:.*]] = "stablehlo.subtract"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%subtract = stablehlo.subtract %t0, %t0 : tensor<i32>

// CHECK: %[[XOR:.*]] = stablehlo.xor %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[XOR:.*]] = "stablehlo.xor"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
%xor = stablehlo.xor %t0, %t0 : tensor<i32>

// Control flow operations

// CHECK: %[[IF_RESULT:.*]] = "stablehlo.if"(%[[PRED]]) ({
// CHECK:   stablehlo.return %[[T0]] : tensor<i32>
// CHECK: }, {
// CHECK:   stablehlo.return %[[T0]] : tensor<i32>
// CHECK: }) : (tensor<i1>) -> tensor<i32>
// CHECK-GENERIC: %[[IF_RESULT:.*]] = "stablehlo.if"(%[[PRED]]) ({
// CHECK-GENERIC:   "stablehlo.return"(%[[T0]]) : (tensor<i32>) -> ()
// CHECK-GENERIC: }, {
// CHECK-GENERIC:   "stablehlo.return"(%[[T0]]) : (tensor<i32>) -> ()
// CHECK-GENERIC: }) : (tensor<i1>) -> tensor<i32>
%if_result = "stablehlo.if"(%pred) ({
  stablehlo.return %t0 : tensor<i32>
}, {
  stablehlo.return %t0 : tensor<i32>
}) : (tensor<i1>) -> tensor<i32>

// CHECK: %[[INIT_I:.*]] = "test.op"() : () -> tensor<i64>
%init_i = "test.op"() : () -> tensor<i64>
// CHECK: %[[INIT_SUM:.*]] = "test.op"() : () -> tensor<i64>
%init_sum = "test.op"() : () -> tensor<i64>
// CHECK: %[[WHILE_RES:.*]] = stablehlo.while(%[[WHILE_ARG0:.*]] = %[[INIT_I]], %[[WHILE_ARG1:.*]] = %[[INIT_SUM]]) : tensor<i64>, tensor<i64> attributes {tag = "loop"}
// CHECK: cond {
// CHECK:   stablehlo.return %[[PRED]] : tensor<i1>
// CHECK: } do {
// CHECK:   stablehlo.return %[[WHILE_ARG0]], %[[WHILE_ARG1]] : tensor<i64>, tensor<i64>
// CHECK: }
// CHECK-GENERIC: %[[WHILE_GEN_RES:.*]] = "stablehlo.while"(%[[WHILE_GEN_INIT_I:.*]], %[[WHILE_GEN_INIT_SUM:.*]]) ({
// CHECK-GENERIC: ^{{.*}}(%[[WHILE_GEN_COND0:.*]]{{ ?}}: tensor<i64>, %[[WHILE_GEN_COND1:.*]]{{ ?}}: tensor<i64>):
// CHECK-GENERIC:   "stablehlo.return"(%[[PRED]]) : (tensor<i1>) -> ()
// CHECK-GENERIC: }, {
// CHECK-GENERIC: ^{{.*}}(%[[WHILE_GEN_BODY0:.*]]{{ ?}}: tensor<i64>, %[[WHILE_GEN_BODY1:.*]]{{ ?}}: tensor<i64>):
// CHECK-GENERIC:   "stablehlo.return"(%[[WHILE_GEN_RET0:.*]], %[[WHILE_GEN_RET1:.*]]) : (tensor<i64>, tensor<i64>) -> ()
// CHECK-GENERIC: }) {tag = "loop"} : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
%while_r0, %while_r1 = stablehlo.while(%while_arg0 = %init_i, %while_arg1 = %init_sum) : tensor<i64>, tensor<i64> attributes {tag = "loop"}
cond {
  stablehlo.return %pred : tensor<i1>
} do {
  stablehlo.return %while_arg0, %while_arg1 : tensor<i64>, tensor<i64>
}

// CHECK: %[[WHILE_ATTR_RES:.*]] = stablehlo.while(%[[WHILE_ATTR_ARG:.*]] = %[[INIT_I]]) : tensor<i64> attributes {tag = "loop"}
// CHECK: cond {
// CHECK:   stablehlo.return %[[PRED]] : tensor<i1>
// CHECK: } do {
// CHECK:   stablehlo.return %[[WHILE_ATTR_ARG]] : tensor<i64>
// CHECK: }
%while_attr = stablehlo.while(%while_arg = %init_i) : tensor<i64> attributes {tag = "loop"}
cond {
  stablehlo.return %pred : tensor<i1>
} do {
  stablehlo.return %while_arg : tensor<i64>
}

// CHECK: stablehlo.while()
// CHECK: cond {
// CHECK:   stablehlo.return %[[PRED]] : tensor<i1>
// CHECK: } do {
// CHECK:   stablehlo.return
// CHECK: }
// CHECK-GENERIC: "stablehlo.while"() ({
// CHECK-GENERIC:   "stablehlo.return"(%[[PRED]]) : (tensor<i1>) -> ()
// CHECK-GENERIC: }, {
// CHECK-GENERIC:   "stablehlo.return"() : () -> ()
// CHECK-GENERIC: }) : () -> ()
stablehlo.while()
cond {
  stablehlo.return %pred : tensor<i1>
} do {
  stablehlo.return
}

// CHECK: %[[OB_RES:.*]] = stablehlo.optimization_barrier %[[T0]], %[[T0]] : tensor<i32>, tensor<i32>
// CHECK-GENERIC: %[[OB_RES:.*]] = "stablehlo.optimization_barrier"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
%ob0, %ob1 = stablehlo.optimization_barrier %t0, %t0 : tensor<i32>, tensor<i32>


// Other operations

// CHECK: %[[TOKEN0:.*]] = "test.op"() : () -> !stablehlo.token
// CHECK-GENERIC: %[[TOKEN0:.*]] = "test.op"() : () -> !stablehlo.token
%token0 = "test.op"() : () -> !stablehlo.token
// CHECK: %[[TOKEN1:.*]] = "test.op"() : () -> !stablehlo.token
// CHECK-GENERIC: %[[TOKEN1:.*]] = "test.op"() : () -> !stablehlo.token
%token1 = "test.op"() : () -> !stablehlo.token
// CHECK: %[[AFTER_ALL:.*]] = stablehlo.after_all %[[TOKEN0]], %[[TOKEN1]] : !stablehlo.token
// CHECK-GENERIC: %[[AFTER_ALL:.*]] = "stablehlo.after_all"(%[[TOKEN0]], %[[TOKEN1]]) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
%after_all = stablehlo.after_all %token0, %token1 : !stablehlo.token

// CHECK: %[[TRANSPOSE_OPERAND:.*]] = "test.op"() : () -> tensor<2x3x2xi32>
// CHECK-GENERIC: %[[TRANSPOSE_OPERAND:.*]] = "test.op"() : () -> tensor<2x3x2xi32>
%transpose_operand = "test.op"() : () -> tensor<2x3x2xi32>
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
// CHECK: %[[TRANSPOSE_RESULT:.*]] = stablehlo.transpose %[[TRANSPOSE_OPERAND]], dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// CHECK-GENERIC: %[[TRANSPOSE_RESULT:.*]] = "stablehlo.transpose"(%[[TRANSPOSE_OPERAND]]) {permutation = array<i64: 2, 1, 0>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
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
//   low = [0, 1], 
//   high = [2, 1], 
//   interior = [1, 2] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// CHECK-GENERIC: %[[PAD_RESULT:.*]] = "stablehlo.pad"(%[[PAD_OPERAND]], %[[PADDING_VALUE]]) {
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

// CHECK: %[[BITCAST:.*]] = stablehlo.bitcast_convert %[[T0]] : (tensor<i32>) -> tensor<2xi16>
// CHECK-GENERIC: %[[BITCAST:.*]] = "stablehlo.bitcast_convert"(%[[T0]]) : (tensor<i32>) -> tensor<2xi16>
%bitcast = stablehlo.bitcast_convert %t0 : (tensor<i32>) -> tensor<2xi16>

// CHECK: %[[INDEX:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[INDEX:.*]] = "test.op"() : () -> tensor<i32>
%index = "test.op"() : () -> tensor<i32>
// CHECK: %[[RESULT_BRANCH0:.*]] = "test.op"() : () -> tensor<2xi64>
// CHECK-GENERIC: %[[RESULT_BRANCH0:.*]] = "test.op"() : () -> tensor<2xi64>
%result_branch0 = "test.op"() : () -> tensor<2xi64>
// CHECK: %[[RESULT_BRANCH1:.*]] = "test.op"() : () -> tensor<2xi64>
// CHECK-GENERIC: %[[RESULT_BRANCH1:.*]] = "test.op"() : () -> tensor<2xi64>
%result_branch1 = "test.op"() : () -> tensor<2xi64>

// CHECK-GENERIC: %[[CASE_RES:.*]] = "stablehlo.case"(%[[INDEX]]) ({
%0:2 = "stablehlo.case"(%index) ({
  // CHECK: stablehlo.return %[[RESULT_BRANCH0]], %[[RESULT_BRANCH0]] : tensor<2xi64>, tensor<2xi64>
  // CHECK-GENERIC: "stablehlo.return"(%[[RESULT_BRANCH0]], %[[RESULT_BRANCH0]]) : (tensor<2xi64>, tensor<2xi64>) -> ()
  stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
}, {
  // CHECK: stablehlo.return %[[RESULT_BRANCH1]], %[[RESULT_BRANCH1]] : tensor<2xi64>, tensor<2xi64>
  // CHECK-GENERIC: "stablehlo.return"(%[[RESULT_BRANCH1]], %[[RESULT_BRANCH1]]) : (tensor<2xi64>, tensor<2xi64>) -> ()
  stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
// CHECK-GENERIC: }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
}) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)

// CHECK: %[[CONSTANT:.*]] = stablehlo.constant dense<{{.*}}> : tensor<2x2xf32>
// CHECK-GENERIC: %[[CONSTANT:.*]] = "stablehlo.constant"() {value = dense<{{.*}}> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
%constant = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>

// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[T0]], %[[T0]], %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[CLAMP:.*]] = "stablehlo.clamp"(%[[T0]], %[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
%clamp = stablehlo.clamp %t0, %t0, %t0 : tensor<i32>

// CHECK: %[[COMPARE:.*]] = stablehlo.compare EQ, %[[T0]], %[[T0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-GENERIC: %[[COMPARE:.*]] = "stablehlo.compare"(%[[T0]], %[[T0]]) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
%compare = stablehlo.compare EQ, %t0, %t0 : (tensor<i32>, tensor<i32>) -> tensor<i1>

// CHECK: %[[MAP:.*]] = "stablehlo.map"(%[[T5F32]], %[[T5F32]]) ({
// CHECK-NEXT: ^bb0(%[[MAP_ARG0:[^ )]+]] : tensor<f32>, %[[MAP_ARG1:[^ )]+]] : tensor<f32>):
// CHECK-NEXT:   %[[MAP_MUL:.*]] = stablehlo.multiply %[[MAP_ARG0]], %[[MAP_ARG1]] : tensor<f32>
// CHECK-NEXT:   stablehlo.return %[[MAP_MUL]] : tensor<f32>
// CHECK-NEXT: }) {dimensions = array<i64: 0>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
%map = "stablehlo.map"(%t5f32, %t5f32) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %result = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    stablehlo.return %result : tensor<f32>
}) {
  dimensions = array<i64: 0>
} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>


// CHECK: %[[REDUCE_PRECISION:.*]] = stablehlo.reduce_precision %[[TF64]], format = e5m10 : tensor<f64>
// CHECK-GENERIC: %[[REDUCE_PRECISION:.*]] = "stablehlo.reduce_precision"(%[[TF64]]) {exponent_bits = 5 : i32, mantissa_bits = 10 : i32} : (tensor<f64>) -> tensor<f64>
%reduce_precision = stablehlo.reduce_precision %tf64, format = e5m10 : tensor<f64>
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
// CHECK-GENERIC:   ^bb[[REDUCE_MULTI_BB:[0-9]+]](%[[REDUCE_GEN_ARG0:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_GEN_ARG2:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_GEN_ARG1:.*]]{{ ?}}: tensor<i64>, %[[REDUCE_GEN_ARG3:.*]]{{ ?}}: tensor<i64>):
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

// CHECK: %[[CUSTOM_CALL_LAYOUTS:.*]] = stablehlo.custom_call @bar(%[[CONSTANT]]) {
// CHECK-SAME: api_version = 4 : i32,
// CHECK-SAME: backend_config = {bar = 42 : i32},
// CHECK-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
// CHECK-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-GENERIC: %[[CUSTOM_CALL_LAYOUTS:.*]] = "stablehlo.custom_call"(%[[CONSTANT]]) <{call_target_name = "bar", api_version = 4 : i32, backend_config = {bar = 42 : i32}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [], has_side_effect = false}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
%custom_call_layouts = stablehlo.custom_call @bar(%constant) {
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
// CHECK-GENERIC: %[[CUSTOM_CALL_TOKEN:.*]] = "stablehlo.custom_call"(%[[TOKEN_INPUT]]) <{call_target_name = "token_passthrough", backend_config = "opaque-config", operand_layouts = [dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [], has_side_effect = false, api_version = 1 : i32}> : (!stablehlo.token) -> !stablehlo.token
%custom_call_token_layout = stablehlo.custom_call @token_passthrough(%token_input) {
  backend_config = "opaque-config",
  operand_layouts = [dense<> : tensor<0xindex>],
  result_layouts = [dense<> : tensor<0xindex>],
  output_operand_aliases = []
} : (!stablehlo.token) -> !stablehlo.token

// CHECK: %[[CUSTOM_CALL_TUPLE:.*]] = stablehlo.custom_call @tuple_result(%[[CONSTANT]]) {
// CHECK-SAME: api_version = 4 : i32,
// CHECK-SAME: backend_config = {bar = 42 : i32},
// CHECK-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
// CHECK-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> tuple<tensor<2x2xf32>, tensor<2x2xf32>>
// CHECK-GENERIC: %[[CUSTOM_CALL_TUPLE:.*]] = "stablehlo.custom_call"(%[[CONSTANT]]) <{call_target_name = "tuple_result", api_version = 4 : i32, backend_config = {bar = 42 : i32}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [], has_side_effect = false}> : (tensor<2x2xf32>) -> tuple<tensor<2x2xf32>, tensor<2x2xf32>>
%custom_call_tuple_result_layouts = stablehlo.custom_call @tuple_result(%constant) {
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

// CHECK: %[[slice_input:.*]] = "test.op"() : () -> tensor<3x8xi64>
%slice_input = "test.op"() : () -> tensor<3x8xi64>

// CHECK: %[[SLICE_RES:.*]] = stablehlo.slice %[[slice_input]] [1:3, 4:8:2] : (tensor<3x8xi64>) -> tensor<2x2xi64>
// CHECK-GENERIC: %[[SLICE_INPUT_GEN:.*]] = "test.op"() : () -> tensor<3x8xi64>
// CHECK-GENERIC: %[[SLICE_RES_GEN:.*]] = "stablehlo.slice"(%[[SLICE_INPUT_GEN]]) <{start_indices = array<i64: 1, 4>, limit_indices = array<i64: 3, 8>, strides = array<i64: 1, 2>}> : (tensor<3x8xi64>) -> tensor<2x2xi64>
%slice = stablehlo.slice %slice_input [1:3, 4:8:2] : (tensor<3x8xi64>) -> tensor<2x2xi64>

// CHECK: %[[SELECT:.*]] = stablehlo.select %[[PRED]], %[[T0]], %[[T0]] : tensor<i1>, tensor<i32>
// CHECK-GENERIC: %[[SELECT:.*]] = "stablehlo.select"(%[[PRED]], %[[T0]], %[[T0]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
%select = stablehlo.select %pred, %t0, %t0 : tensor<i1>, tensor<i32>

// CHECK: %[[SELECT_MISMATCH:.*]] = stablehlo.select %[[PRED]], %[[T0]], %[[T0]] : tensor<i1>, tensor<i32>
// CHECK-GENERIC: %[[SELECT_MISMATCH:.*]] = "stablehlo.select"(%[[PRED]], %[[T0]], %[[T0]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
%select_mismatch = stablehlo.select %pred, %t0, %t0 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>

// CHECK: %[[operand:.*]] = "test.op"() : () -> tensor<1x3xi64>
// CHECK: %[[out_dims:.*]] = "test.op"() : () -> tensor<3xi64>
// CHECK: %[[DYNAMIC_BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim %[[operand]], %[[out_dims]], dims = [2, 1] : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
// CHECK-GENERIC: %[[operand:.*]] = "test.op"() : () -> tensor<1x3xi64>
// CHECK-GENERIC: %[[out_dims:.*]] = "test.op"() : () -> tensor<3xi64>
// CHECK-GENERIC: %[[DYNAMIC_BCAST:.*]] = "stablehlo.dynamic_broadcast_in_dim"(%[[operand]], %[[out_dims]]) <{broadcast_dimensions = array<i64: 2, 1>}> : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
%operand = "test.op"() : () -> tensor<1x3xi64>
%out_dims = "test.op"() : () -> tensor<3xi64>
%dynamic_bcast = stablehlo.dynamic_broadcast_in_dim %operand, %out_dims, dims = [2, 1] : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>

// CHECK: %[[GATHER_INPUT:.*]] = "test.op"() : () -> tensor<2x3x4x2xi32>
%gather_input = "test.op"() : () -> tensor<2x3x4x2xi32>
// CHECK: %[[START_INDICES:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
%start_indices = "test.op"() : () -> tensor<2x2x3x2xi64>
// CHECK: %[[GATHER_RES:.*]] = "stablehlo.gather"(%[[GATHER_INPUT]], %[[START_INDICES]]) <{dimension_numbers = #stablehlo.gather<
// CHECK-NEXT:   offset_dims = [3, 4],
// CHECK-NEXT:   collapsed_slice_dims = [1],
// CHECK-NEXT:   operand_batching_dims = [0],
// CHECK-NEXT:   start_indices_batching_dims = [1],
// CHECK-NEXT:   start_index_map = [2, 1],
// CHECK-NEXT:   index_vector_dim = 3
// CHECK-NEXT: >, slice_sizes = array<i64: 1, 1, 2, 2>, indices_are_sorted = false}> : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
// CHECK-GENERIC: %[[GATHER_INPUT_GEN:.*]] = "test.op"() : () -> tensor<2x3x4x2xi32>
// CHECK-GENERIC: %[[START_INDICES_GEN:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
// CHECK-GENERIC: %[[GATHER_RES_GEN:.*]] = "stablehlo.gather"(%[[GATHER_INPUT_GEN]], %[[START_INDICES_GEN]]) <{
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
// CHECK: %[[SCATTER_RES:.*]] = "stablehlo.scatter"(%[[SCATTER_INPUT]], %[[SCATTER_INDICES]], %[[SCATTER_UPDATES]]) <{scatter_dimension_numbers = #stablehlo.scatter<
// CHECK-NEXT:   update_window_dims = [3, 4],
// CHECK-NEXT:   inserted_window_dims = [1],
// CHECK-NEXT:   input_batching_dims = [0],
// CHECK-NEXT:   scatter_indices_batching_dims = [1],
// CHECK-NEXT:   scatter_dims_to_operand_dims = [2, 1],
// CHECK-NEXT:   index_vector_dim = 3
// CHECK-NEXT: >, indices_are_sorted = false, unique_indices = false}> ({
// CHECK-NEXT: ^bb[[SCATTER_BB:[0-9]+]](%[[SCATTER_ARG0:[^ )]+]] : tensor<i64>, %[[SCATTER_ARG1:[^ )]+]] : tensor<i64>):
// CHECK-NEXT:   %[[SCATTER_ADD:.*]] = stablehlo.add %[[SCATTER_ARG0]], %[[SCATTER_ARG1]] : tensor<i64>
// CHECK-NEXT:   stablehlo.return %[[SCATTER_ADD]] : tensor<i64>
// CHECK-NEXT: }) : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
// CHECK-GENERIC: %[[SCATTER_INPUT_GEN:.*]] = "test.op"() : () -> tensor<2x3x4x2xi64>
// CHECK-GENERIC: %[[SCATTER_INDICES_GEN:.*]] = "test.op"() : () -> tensor<2x2x3x2xi64>
// CHECK-GENERIC: %[[SCATTER_UPDATES_GEN:.*]] = "test.op"() : () -> tensor<2x2x3x2x2xi64>
// CHECK-GENERIC: %[[SCATTER_RES_GEN:.*]] = "stablehlo.scatter"(%[[SCATTER_INPUT_GEN]], %[[SCATTER_INDICES_GEN]], %[[SCATTER_UPDATES_GEN]]) <{
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

// CHECK: %[[SELECT_FUNCTION_TYPE:.*]] = stablehlo.select %[[PRED]], %[[T0]], %[[T0]] : tensor<i1>, tensor<i32>
// CHECK-GENERIC: %[[SELECT_FUNCTION_TYPE:.*]] = "stablehlo.select"(%[[PRED]], %[[T0]], %[[T0]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
%select_function_type = stablehlo.select %pred, %t0, %t0 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>

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

// CHECK: %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<4x5xi32>
// CHECK-GENERIC: %[[IOTA:.*]] = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4x5xi32>
%iota = stablehlo.iota dim = 0 : tensor<4x5xi32>
