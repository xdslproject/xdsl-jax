// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
%t0 = "test.op"() : () -> tensor<i32>
// CHECK: %[[TF32:.*]] = "test.op"() : () -> tensor<f32>
// CHECK-GENERIC: %[[TF32:.*]] = "test.op"() : () -> tensor<f32>
%tf32 = "test.op"() : () -> tensor<f32>
// CHECK: %[[TCOMPLEX:.*]] = "test.op"() : () -> tensor<complex<f32>>
// CHECK-GENERIC: %[[TCOMPLEX:.*]] = "test.op"() : () -> tensor<complex<f32>>
%tcomplex = "test.op"() : () -> tensor<complex<f32>>

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

// CHECK: %[[FLOOR:.*]] = stablehlo.floor %[[TF32]] : tensor<f32>
// CHECK-GENERIC: %[[FLOOR:.*]] = "stablehlo.floor"(%[[TF32]]) : (tensor<f32>) -> tensor<f32>
%floor = stablehlo.floor %tf32 : tensor<f32>

// CHECK: %[[IMAG:.*]] = stablehlo.imag %[[TCOMPLEX]] : (tensor<complex<f32>>) -> tensor<f32>
// CHECK-GENERIC: %[[IMAG:.*]] = "stablehlo.imag"(%[[TCOMPLEX]]) : (tensor<complex<f32>>) -> tensor<f32>
%imag = stablehlo.imag %tcomplex : (tensor<complex<f32>>) -> tensor<f32>

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

// CHECK: %[[NEGATE:.*]] = stablehlo.negate %[[T0]] : tensor<i32>
// CHECK-GENERIC: %[[NEGATE:.*]] = "stablehlo.negate"(%[[T0]]) : (tensor<i32>) -> tensor<i32>
%negate = stablehlo.negate %t0 : tensor<i32>

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
