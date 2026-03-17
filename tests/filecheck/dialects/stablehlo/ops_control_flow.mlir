// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: JAX_ROUNDTRIP
// RUN: JAX_GENERIC_ROUNDTRIP
// RUN: XDSL_JAX_ROUNDTRIP
// RUN: XDSL_JAX_GENERIC_ROUNDTRIP

// CHECK: %[[PRED:.*]] = "test.op"() : () -> tensor<i1>
// CHECK-GENERIC: %[[PRED:.*]] = "test.op"() : () -> tensor<i1>
%pred = "test.op"() : () -> tensor<i1>
// CHECK: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
// CHECK-GENERIC: %[[T0:.*]] = "test.op"() : () -> tensor<i32>
%t0 = "test.op"() : () -> tensor<i32>

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

// CHECK: %[[TOKEN0:.*]] = "test.op"() : () -> !stablehlo.token
// CHECK-GENERIC: %[[TOKEN0:.*]] = "test.op"() : () -> !stablehlo.token
%token0 = "test.op"() : () -> !stablehlo.token
// CHECK: %[[TOKEN1:.*]] = "test.op"() : () -> !stablehlo.token
// CHECK-GENERIC: %[[TOKEN1:.*]] = "test.op"() : () -> !stablehlo.token
%token1 = "test.op"() : () -> !stablehlo.token
// CHECK: %[[AFTER_ALL:.*]] = stablehlo.after_all %[[TOKEN0]], %[[TOKEN1]] : !stablehlo.token
// CHECK-GENERIC: %[[AFTER_ALL:.*]] = "stablehlo.after_all"(%[[TOKEN0]], %[[TOKEN1]]) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
%after_all = stablehlo.after_all %token0, %token1 : !stablehlo.token

// Optimization barrier.
// CHECK: %[[OB_RES:.*]] = stablehlo.optimization_barrier %[[T0]], %[[T0]] : tensor<i32>, tensor<i32>
// CHECK-GENERIC: %[[OB_RES:.*]] = "stablehlo.optimization_barrier"(%[[T0]], %[[T0]]) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
%ob0, %ob1 = stablehlo.optimization_barrier %t0, %t0 : tensor<i32>, tensor<i32>
