// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect operands to be compatible with condition block arguments
%while_bad_cond_args = "stablehlo.while"(%while_init) ({
^bb0(%arg0: tensor<i32>):
  %pred = "test.op"() : () -> tensor<i1>
  "stablehlo.return"(%pred) : (tensor<i1>) -> ()
}, {
^bb0(%arg0: tensor<i64>):
  "stablehlo.return"(%arg0) : (tensor<i64>) -> ()
}) : (tensor<i64>) -> tensor<i64>

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect operands to be compatible with body block arguments
%while_bad_body_args = "stablehlo.while"(%while_init) ({
^bb0(%arg0: tensor<i64>):
  %pred = "test.op"() : () -> tensor<i1>
  "stablehlo.return"(%pred) : (tensor<i1>) -> ()
}, {
^bb0(%arg0: tensor<i32>):
  %body = "test.op"() : () -> tensor<i64>
  "stablehlo.return"(%body) : (tensor<i64>) -> ()
}) : (tensor<i64>) -> tensor<i64>

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect operands to be compatible with body block return types
%while_bad_body_return = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %pred = "test.op"() : () -> tensor<i1>
  stablehlo.return %pred : tensor<i1>
} do {
  %body = "test.op"() : () -> tensor<i32>
  stablehlo.return %body : tensor<i32>
}

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect condition body returns a single value but got 2
%while_bad_cond_return_count = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %pred = "test.op"() : () -> tensor<i1>
  stablehlo.return %pred, %pred : tensor<i1>, tensor<i1>
} do {
  stablehlo.return %arg0 : tensor<i64>
}

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect condition block return a zero-ranked tensor of i1 but got tensor<1xi1>
%while_bad_cond_return_type = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %pred = "test.op"() : () -> tensor<1xi1>
  stablehlo.return %pred : tensor<1xi1>
} do {
  stablehlo.return %arg0 : tensor<i64>
}

// -----

%while_init = "test.op"() : () -> tensor<i64>
// CHECK: Operation does not verify: expect condition block return a zero-ranked tensor of i1 but got !stablehlo.token
%while_bad_cond_return_token = stablehlo.while(%arg0 = %while_init) : tensor<i64>
cond {
  %tok = "test.op"() : () -> !stablehlo.token
  stablehlo.return %tok : !stablehlo.token
} do {
  stablehlo.return %arg0 : tensor<i64>
}

// -----

%arg = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: requires the same number of operands and results (got 1 operands and 2 results)
%bad0, %bad1 = "stablehlo.optimization_barrier"(%arg) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)

// -----

%arg = "test.op"() : () -> tensor<i32>
// CHECK: Operation does not verify: requires the same type for operand and result at index 0 (got tensor<i32> vs tensor<i1>)
%bad = "stablehlo.optimization_barrier"(%arg) : (tensor<i32>) -> tensor<i1>
