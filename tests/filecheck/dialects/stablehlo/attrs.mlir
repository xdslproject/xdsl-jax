// RUN: XDSL_ROUNDTRIP

// CHECK:       builtin.module {

// CHECK-NEXT:    "test.op"() {default = #stablehlo<precision DEFAULT>, high = #stablehlo<precision HIGH>, highest = #stablehlo<precision HIGHEST>} : () -> ()
"test.op"() {
    default = #stablehlo<precision DEFAULT>,
    high = #stablehlo<precision HIGH>,
    highest = #stablehlo<precision HIGHEST>
} : () -> ()

// CHECK-NEXT:    %token = "test.op"() : () -> !stablehlo.token
%token = "test.op"() : () -> (!stablehlo.token)

// CHECK-NEXT:    "test.op"() {dot = #stablehlo.dot<
// CHECK-NEXT:      lhs_batching_dimensions = [0],
// CHECK-NEXT:      rhs_batching_dimensions = [1],
// CHECK-NEXT:      lhs_contracting_dimensions = [2],
// CHECK-NEXT:      rhs_contracting_dimensions = [3]
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [1],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [3]
    >
} : () -> ()

// CHECK-NEXT:    "test.op"() {dot = #stablehlo.dot<
// CHECK-NEXT:      lhs_contracting_dimensions = [0],
// CHECK-NEXT:      rhs_contracting_dimensions = [1]
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_contracting_dimensions = [0],
        rhs_contracting_dimensions = [1]
    >
} : () -> ()

// CHECK-NEXT:    "test.op"() {
// CHECK-SAME:      eq = #stablehlo<comparison_direction EQ>,
// CHECK-SAME:      ne = #stablehlo<comparison_direction NE>,
// CHECK-SAME:      ge = #stablehlo<comparison_direction GE>,
// CHECK-SAME:      gt = #stablehlo<comparison_direction GT>,
// CHECK-SAME:      le = #stablehlo<comparison_direction LE>,
// CHECK-SAME:      lt = #stablehlo<comparison_direction LT>
// CHECK-SAME:     } : () -> ()
"test.op"() {
  eq = #stablehlo<comparison_direction EQ>,
  ne = #stablehlo<comparison_direction NE>,
  ge = #stablehlo<comparison_direction GE>,
  gt = #stablehlo<comparison_direction GT>,
  le = #stablehlo<comparison_direction LE>,
  lt = #stablehlo<comparison_direction LT>
} : () -> ()

// CHECK-NEXT:    "test.op"() {
// CHECK-SAME:      notype = #stablehlo<comparison_type NOTYPE>,
// CHECK-SAME:      float = #stablehlo<comparison_type FLOAT>,
// CHECK-SAME:      totalorder = #stablehlo<comparison_type TOTALORDER>,
// CHECK-SAME:      signed = #stablehlo<comparison_type SIGNED>,
// CHECK-SAME:      unsigned = #stablehlo<comparison_type UNSIGNED>
// CHECK-SAME:    } : () -> ()
"test.op"() {
  notype = #stablehlo<comparison_type NOTYPE>,
  float = #stablehlo<comparison_type FLOAT>,
  totalorder = #stablehlo<comparison_type TOTALORDER>,
  signed = #stablehlo<comparison_type SIGNED>,
  unsigned = #stablehlo<comparison_type UNSIGNED>
} : () -> ()

// CHECK-NEXT:    "test.op"() {
// CHECK-SAME:      default = #stablehlo<result_accuracy_mode DEFAULT>,
// CHECK-SAME:      high = #stablehlo<result_accuracy_mode HIGHEST>,
// CHECK-SAME:      highest = #stablehlo<result_accuracy_mode TOLERANCE>
// CHECK-SAME:    } : () -> ()
"test.op"() {
  default = #stablehlo<result_accuracy_mode DEFAULT>,
  high = #stablehlo<result_accuracy_mode HIGHEST>,
  highest = #stablehlo<result_accuracy_mode TOLERANCE>
} : () -> ()

// CHECK-NEXT:    "test.op"() {gather = #stablehlo.gather<
// CHECK-NEXT:      offset_dims = [0, 1],
// CHECK-NEXT:      collapsed_slice_dims = [2],
// CHECK-NEXT:      operand_batching_dims = [3],
// CHECK-NEXT:      start_indices_batching_dims = [4],
// CHECK-NEXT:      start_index_map = [5],
// CHECK-NEXT:      index_vector_dim = 6
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  gather = #stablehlo.gather<
    offset_dims = [0, 1],
    collapsed_slice_dims = [2],
    operand_batching_dims = [3],
    start_indices_batching_dims = [4],
    start_index_map = [5],
    index_vector_dim = 6
  >
} : () -> ()

// CHECK-NEXT:    "test.op"() {scatter = #stablehlo.scatter<
// CHECK-NEXT:      update_window_dims = [0, 1],
// CHECK-NEXT:      inserted_window_dims = [2],
// CHECK-NEXT:      input_batching_dims = [3],
// CHECK-NEXT:      scatter_indices_batching_dims = [4],
// CHECK-NEXT:      scatter_dims_to_operand_dims = [5],
// CHECK-NEXT:      index_vector_dim = 6
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  scatter = #stablehlo.scatter<
    update_window_dims = [0, 1],
    inserted_window_dims = [2],
    input_batching_dims = [3],
    scatter_indices_batching_dims = [4],
    scatter_dims_to_operand_dims = [5],
    index_vector_dim = 6
  >
} : () -> ()

// CHECK-NEXT:    "test.op"() {
// CHECK-SAME:      unspecified = #stablehlo<custom_call_api_version API_VERSION_UNSPECIFIED>,
// CHECK-SAME:      original = #stablehlo<custom_call_api_version API_VERSION_ORIGINAL>,
// CHECK-SAME:      status_returning = #stablehlo<custom_call_api_version API_VERSION_STATUS_RETURNING>,
// CHECK-SAME:      status_returning_unified = #stablehlo<custom_call_api_version API_VERSION_STATUS_RETURNING_UNIFIED>,
// CHECK-SAME:      typed_ffi = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>
// CHECK-SAME:    } : () -> ()
"test.op"() {
  unspecified = #stablehlo<custom_call_api_version API_VERSION_UNSPECIFIED>,
  original = #stablehlo<custom_call_api_version API_VERSION_ORIGINAL>,
  status_returning = #stablehlo<custom_call_api_version API_VERSION_STATUS_RETURNING>,
  status_returning_unified = #stablehlo<custom_call_api_version API_VERSION_STATUS_RETURNING_UNIFIED>,
  typed_ffi = #stablehlo<custom_call_api_version API_VERSION_TYPED_FFI>
} : () -> ()

// CHECK-NEXT:    "test.op"() {alias = #stablehlo.output_operand_alias<
// CHECK-NEXT:      output_tuple_indices = [0],
// CHECK-NEXT:      operand_index = 1,
// CHECK-NEXT:      operand_tuple_indices = [2]
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  alias = #stablehlo.output_operand_alias<
    output_tuple_indices = [0],
    operand_index = 1,
    operand_tuple_indices = [2]
  >
} : () -> ()

// CHECK-NEXT:  }
