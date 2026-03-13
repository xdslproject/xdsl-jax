// RUN: XDSL_ROUNDTRIP

// CHECK:       builtin.module {

// === Precision ===
// CHECK-NEXT:    "test.op"() {default = #stablehlo<precision DEFAULT>, high = #stablehlo<precision HIGH>, highest = #stablehlo<precision HIGHEST>} : () -> ()
"test.op"() {
    default = #stablehlo<precision DEFAULT>,
    high = #stablehlo<precision HIGH>,
    highest = #stablehlo<precision HIGHEST>
} : () -> ()

// === Token Type ===
// CHECK-NEXT:    %token = "test.op"() : () -> !stablehlo.token
%token = "test.op"() : () -> (!stablehlo.token)

// === Dot Dimension Numbers ===
// CHECK-NEXT:    "test.op"() {dot = #stablehlo.dot<
// CHECK-SAME:      lhs_batching_dimensions = [0],
// CHECK-SAME:      rhs_batching_dimensions = [1],
// CHECK-SAME:      lhs_contracting_dimensions = [2],
// CHECK-SAME:      rhs_contracting_dimensions = [3]
// CHECK-SAME:    >} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [1],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [3]
    >
} : () -> ()

// CHECK-NEXT: "test.op"() {dot = #stablehlo.dot<>} : () -> ()
"test.op"() {dot = #stablehlo.dot<>} : () -> ()

// CHECK-NEXT: "test.op"() {dot = #stablehlo.dot<>} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [],
        rhs_contracting_dimensions = []
    >
} : () -> ()

// CHECK-NEXT:    "test.op"() {dot = #stablehlo.dot<
// CHECK-SAME:      rhs_batching_dimensions = [0],
// CHECK-SAME:      rhs_contracting_dimensions = [1]
// CHECK-SAME:    >} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [],
        rhs_contracting_dimensions = [1]
    >
} : () -> ()

// CHECK-NEXT:    "test.op"() {dot = #stablehlo.dot<
// CHECK-SAME:      lhs_batching_dimensions = [0],
// CHECK-SAME:      rhs_contracting_dimensions = [1]
// CHECK-SAME:    >} : () -> ()
"test.op"() {
    dot = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_contracting_dimensions = [1]
    >
} : () -> ()

// === Comparison Direction ===
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

// === Comparison Type ===
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

// === Result Accuracy Mode ===
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

// === Output Operand Alias ===
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

// CHECK-NEXT:    "test.op"() {alias_empty = #stablehlo.output_operand_alias<
// CHECK-NEXT:      output_tuple_indices = [],
// CHECK-NEXT:      operand_index = 0,
// CHECK-NEXT:      operand_tuple_indices = []
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  alias_empty = #stablehlo.output_operand_alias<
    output_tuple_indices = [],
    operand_index = 0,
    operand_tuple_indices = []
  >
} : () -> ()

// === Scatter Dimension Numbers ===
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

// === Gather Dimension Numbers ===
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

// CHECK-NEXT:    "test.op"() {gather_reordered = #stablehlo.gather<
// CHECK-NEXT:      offset_dims = [2],
// CHECK-NEXT:      collapsed_slice_dims = [1],
// CHECK-NEXT:      operand_batching_dims = [0],
// CHECK-NEXT:      start_indices_batching_dims = [1],
// CHECK-NEXT:      start_index_map = [1],
// CHECK-NEXT:      index_vector_dim = 2
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  gather_reordered = #stablehlo.gather<
    collapsed_slice_dims = [1],
    operand_batching_dims = [0],
    start_indices_batching_dims = [1],
    index_vector_dim = 2,
    offset_dims = [2],
    start_index_map = [1]
  >
} : () -> ()

// CHECK-NEXT:    "test.op"() {gather_minimal = #stablehlo.gather<
// CHECK-NEXT:      offset_dims = [0],
// CHECK-NEXT:      index_vector_dim = 1
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  gather_minimal = #stablehlo.gather<
    offset_dims = [0],
    index_vector_dim = 1
  >
} : () -> ()

// CHECK-NEXT:    "test.op"() {gather_defaults = #stablehlo.gather<
// CHECK-NEXT:      offset_dims = [0],
// CHECK-NEXT:      start_index_map = [1]
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  gather_defaults = #stablehlo.gather<
    offset_dims = [0],
    collapsed_slice_dims = [],
    operand_batching_dims = [],
    start_indices_batching_dims = [],
    start_index_map = [1],
    index_vector_dim = 0
  >
} : () -> ()

// CHECK-NEXT:    "test.op"() {gather_trailing_comma = #stablehlo.gather<
// CHECK-NEXT:      offset_dims = [0],
// CHECK-NEXT:      start_index_map = [1],
// CHECK-NEXT:      index_vector_dim = 2
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  gather_trailing_comma = #stablehlo.gather<
    offset_dims = [0],
    start_index_map = [1],
    index_vector_dim = 2,
  >
} : () -> ()

// CHECK-NEXT:    "test.op"() {dot_algorithm = #stablehlo.dot_algorithm<
// CHECK-NEXT:      lhs_precision_type = f32,
// CHECK-NEXT:      rhs_precision_type = f32,
// CHECK-NEXT:      accumulation_type = f32,
// CHECK-NEXT:      lhs_component_count = 1,
// CHECK-NEXT:      rhs_component_count = 1,
// CHECK-NEXT:      num_primitive_operations = 1,
// CHECK-NEXT:      allow_imprecise_accumulation = false
// CHECK-NEXT:    >} : () -> ()
"test.op"() {
  dot_algorithm = #stablehlo.dot_algorithm<
    lhs_precision_type = f32,
    rhs_precision_type = f32,
    accumulation_type = f32,
    lhs_component_count = 1,
    rhs_component_count = 1,
    num_primitive_operations = 1,
    allow_imprecise_accumulation = false
  >
} : () -> ()
