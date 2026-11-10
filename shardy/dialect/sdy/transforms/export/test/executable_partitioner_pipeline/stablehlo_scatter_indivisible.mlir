// RUN: %S/run_sdy_interpreter_test.sh %s %t --partition_count=4

//--- part1.mlir

sdy.mesh @mesh_x2_y2 = <["x"=2, "y"=2]>

// Scatter min where scatter_indices (5x1xi32) and updates (5x2xi32) are
// unevenly sharded across 4 partitions (padded 5 -> 8). Indices [1..5] never
// touch row 0, so row 0 must remain [100, 101] rather than min(100, 0) == 0.
func.func @parallel_scatter_min_indivisible_indices(
    %arg0: tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg1: tensor<5x1xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg2: tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
    -> (tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %idx = sdy.reshard %arg1 <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<5x1xi32>
  %upd = sdy.reshard %arg2 <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<5x2xi32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %upd) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %min = stablehlo.minimum %lhs, %rhs : tensor<i32>
    stablehlo.return %min : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{}, {}]>]>
  } : (tensor<6x2xi32>, tensor<5x1xi32>, tensor<5x2xi32>) -> tensor<6x2xi32>
  return %0 : tensor<6x2xi32>
}

func.func @sequential_scatter_min_indivisible_indices(
    %arg0: tensor<6x2xi32>, %arg1: tensor<5x1xi32>, %arg2: tensor<5x2xi32>)
    -> tensor<6x2xi32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %min = stablehlo.minimum %lhs, %rhs : tensor<i32>
    stablehlo.return %min : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<6x2xi32>, tensor<5x1xi32>, tensor<5x2xi32>) -> tensor<6x2xi32>
  return %0 : tensor<6x2xi32>
}

// Scatter add where updates are produced by an elementwise op (upd + 1.0) so
// the padded update slots [5..7] hold 1.0 rather than 0.0. Padded indices must
// still be masked out of bounds so row 0 does not accumulate +3.0.
func.func @parallel_scatter_add_nonzero_padded_updates(
    %arg0: tensor<6x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg1: tensor<5x1xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg2: tensor<5x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
    -> (tensor<6x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %idx = sdy.reshard %arg1 <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<5x1xi32>
  %upd_sharded = sdy.reshard %arg2 <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<5x2xf32>
  %one = stablehlo.constant dense<1.000000e+00> : tensor<5x2xf32>
  %one_sharded = sdy.reshard %one <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<5x2xf32>
  %upd = stablehlo.add %upd_sharded, %one_sharded {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x", "y"}, {}]>]>} : tensor<5x2xf32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %upd) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{}, {}], unreduced={"x", "y"}>]>
  } : (tensor<6x2xf32>, tensor<5x1xi32>, tensor<5x2xf32>) -> tensor<6x2xf32>
  %1 = sdy.reshard %0 <@mesh_x2_y2, [{}, {}]> : tensor<6x2xf32>
  return %1 : tensor<6x2xf32>
}

func.func @sequential_scatter_add_nonzero_padded_updates(
    %arg0: tensor<6x2xf32>, %arg1: tensor<5x1xi32>, %arg2: tensor<5x2xf32>)
    -> tensor<6x2xf32> {
  %one = stablehlo.constant dense<1.000000e+00> : tensor<5x2xf32>
  %upd = stablehlo.add %arg2, %one : tensor<5x2xf32>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %upd) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<6x2xf32>, tensor<5x1xi32>, tensor<5x2xf32>) -> tensor<6x2xf32>
  return %0 : tensor<6x2xf32>
}

// Scatter max with unsigned ui32 scatter_indices sharded on "y"=2 (padded 5->6)
// and operand sharded on "x"=2 (local shard size 3, shard 1 offset 3), testing
// both UINT32_MAX sentinel generation and local index offset subtraction.
func.func @parallel_scatter_max_ui32_sharded_operand_and_indices(
    %arg0: tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg1: tensor<5x1xui32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg2: tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
    -> (tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %in = sdy.reshard %arg0 <@mesh_x2_y2, [{"x"}, {}]> : tensor<6x2xi32>
  %idx = sdy.reshard %arg1 <@mesh_x2_y2, [{"y"}, {}]> : tensor<5x1xui32>
  %upd = sdy.reshard %arg2 <@mesh_x2_y2, [{"y"}, {}]> : tensor<5x2xi32>
  %0 = "stablehlo.scatter"(%in, %idx, %upd) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %max = stablehlo.maximum %lhs, %rhs : tensor<i32>
    stablehlo.return %max : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {}]>]>
  } : (tensor<6x2xi32>, tensor<5x1xui32>, tensor<5x2xi32>) -> tensor<6x2xi32>
  %1 = sdy.reshard %0 <@mesh_x2_y2, [{}, {}]> : tensor<6x2xi32>
  return %1 : tensor<6x2xi32>
}

func.func @sequential_scatter_max_ui32_sharded_operand_and_indices(
    %arg0: tensor<6x2xi32>, %arg1: tensor<5x1xui32>, %arg2: tensor<5x2xi32>)
    -> tensor<6x2xi32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %max = stablehlo.maximum %lhs, %rhs : tensor<i32>
    stablehlo.return %max : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<6x2xi32>, tensor<5x1xui32>, tensor<5x2xi32>) -> tensor<6x2xi32>
  return %0 : tensor<6x2xi32>
}

// Scatter min with narrow ui8 and i8 scatter_indices sharded on "y"=2 (padded
// 5 -> 6) and operand sharded on "x"=2 (local shard size 3, shard 1 offset 3).
func.func @parallel_scatter_min_ui8_sharded_operand_and_indices(
    %arg0: tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg1: tensor<5x1xui8> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg2: tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
    -> (tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %in = sdy.reshard %arg0 <@mesh_x2_y2, [{"x"}, {}]> : tensor<6x2xi32>
  %idx = sdy.reshard %arg1 <@mesh_x2_y2, [{"y"}, {}]> : tensor<5x1xui8>
  %upd = sdy.reshard %arg2 <@mesh_x2_y2, [{"y"}, {}]> : tensor<5x2xi32>
  %0 = "stablehlo.scatter"(%in, %idx, %upd) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %min = stablehlo.minimum %lhs, %rhs : tensor<i32>
    stablehlo.return %min : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {}]>]>
  } : (tensor<6x2xi32>, tensor<5x1xui8>, tensor<5x2xi32>) -> tensor<6x2xi32>
  %1 = sdy.reshard %0 <@mesh_x2_y2, [{}, {}]> : tensor<6x2xi32>
  return %1 : tensor<6x2xi32>
}

func.func @sequential_scatter_min_ui8_sharded_operand_and_indices(
    %arg0: tensor<6x2xi32>, %arg1: tensor<5x1xui8>, %arg2: tensor<5x2xi32>)
    -> tensor<6x2xi32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %min = stablehlo.minimum %lhs, %rhs : tensor<i32>
    stablehlo.return %min : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<6x2xi32>, tensor<5x1xui8>, tensor<5x2xi32>) -> tensor<6x2xi32>
  return %0 : tensor<6x2xi32>
}

// Scatter with indivisible sharded ui8 scatter_indices (3x1xui8 sharded by 2,
// padded to 4x1) into a 300-element operand (> 255), verifying that -1 sentinel
// padding promotes ui8 indices instead of scattering into index 255.
func.func @parallel_scatter_indivisible_ui8_indices_large_operand(
    %arg0: tensor<300xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>},
    %arg1: tensor<3x1xui8> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg2: tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>})
    -> (tensor<300xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>}) {
  %idx = sdy.reshard %arg1 <@mesh_x2_y2, [{"x"}, {}]> : tensor<3x1xui8>
  %upd = sdy.reshard %arg2 <@mesh_x2_y2, [{"x"}]> : tensor<3xi32>
  %0 = "stablehlo.scatter"(%arg0, %idx, %upd) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{}]>]>
  } : (tensor<300xi32>, tensor<3x1xui8>, tensor<3xi32>) -> tensor<300xi32>
  return %0 : tensor<300xi32>
}

func.func @sequential_scatter_indivisible_ui8_indices_large_operand(
    %arg0: tensor<300xi32>, %arg1: tensor<3x1xui8>, %arg2: tensor<3xi32>)
    -> tensor<300xi32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %sum = stablehlo.add %lhs, %rhs : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>
  } : (tensor<300xi32>, tensor<3x1xui8>, tensor<3xi32>) -> tensor<300xi32>
  return %0 : tensor<300xi32>
}

// Dynamic update slice with narrow i8 scalar start index on a sharded 150-elem
// operand (padded to 152), testing clampScalarStartIndex and
// rewriteSingleShardUpdate narrow index promotion.
func.func @parallel_dynamic_update_slice_i8_large_operand(
    %arg0: tensor<150xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>},
    %arg1: tensor<1xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>},
    %arg2: tensor<i8> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, []>})
    -> (tensor<150xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>}) {
  %in = sdy.reshard %arg0 <@mesh_x2_y2, [{"x"}]> : tensor<150xi32>
  %0 = stablehlo.dynamic_update_slice %in, %arg1, %arg2 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}]>]>
  } : (tensor<150xi32>, tensor<1xi32>, tensor<i8>) -> tensor<150xi32>
  %1 = sdy.reshard %0 <@mesh_x2_y2, [{}]> : tensor<150xi32>
  return %1 : tensor<150xi32>
}

func.func @sequential_dynamic_update_slice_i8_large_operand(
    %arg0: tensor<150xi32>, %arg1: tensor<1xi32>, %arg2: tensor<i8>)
    -> tensor<150xi32> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2
      : (tensor<150xi32>, tensor<1xi32>, tensor<i8>) -> tensor<150xi32>
  return %0 : tensor<150xi32>
}

//--- part2.mlir

func.func @main() {
  %in_pos = stablehlo.constant dense<[
    [100, 101], [200, 201], [300, 301], [400, 401], [500, 501], [600, 601]
  ]> : tensor<6x2xi32>
  %idx_i32 = stablehlo.constant dense<[[1], [2], [3], [4], [5]]> : tensor<5x1xi32>
  %upd_pos = stablehlo.constant dense<[
    [10, 11], [20, 21], [30, 31], [40, 41], [50, 51]
  ]> : tensor<5x2xi32>

  %seq_min = func.call @sequential_scatter_min_indivisible_indices(%in_pos, %idx_i32, %upd_pos)
      : (tensor<6x2xi32>, tensor<5x1xi32>, tensor<5x2xi32>) -> tensor<6x2xi32>
  %res_min:4 = "interpreter.run_parallel"(
      %in_pos, %idx_i32, %upd_pos,
      %in_pos, %idx_i32, %upd_pos,
      %in_pos, %idx_i32, %upd_pos,
      %in_pos, %idx_i32, %upd_pos) {
    programs = [[
      @parallel_scatter_min_indivisible_indices,
      @parallel_scatter_min_indivisible_indices,
      @parallel_scatter_min_indivisible_indices,
      @parallel_scatter_min_indivisible_indices
    ]]
  } : (tensor<6x2xi32>, tensor<5x1xi32>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xi32>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xi32>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xi32>, tensor<5x2xi32>)
      -> (tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>)
  "check.expect_eq"(%res_min#0, %seq_min) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_min#1, %seq_min) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_min#2, %seq_min) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_min#3, %seq_min) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()

  %in_f32 = stablehlo.constant dense<[
    [100.0, 101.0], [200.0, 201.0], [300.0, 301.0],
    [400.0, 401.0], [500.0, 501.0], [600.0, 601.0]
  ]> : tensor<6x2xf32>
  %upd_f32 = stablehlo.constant dense<[
    [10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0], [50.0, 51.0]
  ]> : tensor<5x2xf32>

  %seq_add = func.call @sequential_scatter_add_nonzero_padded_updates(%in_f32, %idx_i32, %upd_f32)
      : (tensor<6x2xf32>, tensor<5x1xi32>, tensor<5x2xf32>) -> tensor<6x2xf32>
  %res_add:4 = "interpreter.run_parallel"(
      %in_f32, %idx_i32, %upd_f32,
      %in_f32, %idx_i32, %upd_f32,
      %in_f32, %idx_i32, %upd_f32,
      %in_f32, %idx_i32, %upd_f32) {
    programs = [[
      @parallel_scatter_add_nonzero_padded_updates,
      @parallel_scatter_add_nonzero_padded_updates,
      @parallel_scatter_add_nonzero_padded_updates,
      @parallel_scatter_add_nonzero_padded_updates
    ]]
  } : (tensor<6x2xf32>, tensor<5x1xi32>, tensor<5x2xf32>,
       tensor<6x2xf32>, tensor<5x1xi32>, tensor<5x2xf32>,
       tensor<6x2xf32>, tensor<5x1xi32>, tensor<5x2xf32>,
       tensor<6x2xf32>, tensor<5x1xi32>, tensor<5x2xf32>)
      -> (tensor<6x2xf32>, tensor<6x2xf32>, tensor<6x2xf32>, tensor<6x2xf32>)
  "check.expect_eq"(%res_add#0, %seq_add) : (tensor<6x2xf32>, tensor<6x2xf32>) -> ()
  "check.expect_eq"(%res_add#1, %seq_add) : (tensor<6x2xf32>, tensor<6x2xf32>) -> ()
  "check.expect_eq"(%res_add#2, %seq_add) : (tensor<6x2xf32>, tensor<6x2xf32>) -> ()
  "check.expect_eq"(%res_add#3, %seq_add) : (tensor<6x2xf32>, tensor<6x2xf32>) -> ()

  %in_neg = stablehlo.constant dense<[
    [-100, -101], [-200, -201], [-300, -301], [-400, -401], [-500, -501], [-600, -601]
  ]> : tensor<6x2xi32>
  %idx_ui32 = stablehlo.constant dense<[[1], [2], [3], [4], [5]]> : tensor<5x1xui32>
  %upd_neg = stablehlo.constant dense<[
    [-10, -11], [-20, -21], [-30, -31], [-40, -41], [-50, -51]
  ]> : tensor<5x2xi32>

  %seq_max = func.call @sequential_scatter_max_ui32_sharded_operand_and_indices(%in_neg, %idx_ui32, %upd_neg)
      : (tensor<6x2xi32>, tensor<5x1xui32>, tensor<5x2xi32>) -> tensor<6x2xi32>
  %res_max:4 = "interpreter.run_parallel"(
      %in_neg, %idx_ui32, %upd_neg,
      %in_neg, %idx_ui32, %upd_neg,
      %in_neg, %idx_ui32, %upd_neg,
      %in_neg, %idx_ui32, %upd_neg) {
    programs = [[
      @parallel_scatter_max_ui32_sharded_operand_and_indices,
      @parallel_scatter_max_ui32_sharded_operand_and_indices,
      @parallel_scatter_max_ui32_sharded_operand_and_indices,
      @parallel_scatter_max_ui32_sharded_operand_and_indices
    ]]
  } : (tensor<6x2xi32>, tensor<5x1xui32>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xui32>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xui32>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xui32>, tensor<5x2xi32>)
      -> (tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>)
  "check.expect_eq"(%res_max#0, %seq_max) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_max#1, %seq_max) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_max#2, %seq_max) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_max#3, %seq_max) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()

  %idx_ui8 = stablehlo.constant dense<[[1], [2], [3], [4], [5]]> : tensor<5x1xui8>
  %seq_ui8 = func.call @sequential_scatter_min_ui8_sharded_operand_and_indices(%in_pos, %idx_ui8, %upd_pos)
      : (tensor<6x2xi32>, tensor<5x1xui8>, tensor<5x2xi32>) -> tensor<6x2xi32>
  %res_ui8:4 = "interpreter.run_parallel"(
      %in_pos, %idx_ui8, %upd_pos,
      %in_pos, %idx_ui8, %upd_pos,
      %in_pos, %idx_ui8, %upd_pos,
      %in_pos, %idx_ui8, %upd_pos) {
    programs = [[
      @parallel_scatter_min_ui8_sharded_operand_and_indices,
      @parallel_scatter_min_ui8_sharded_operand_and_indices,
      @parallel_scatter_min_ui8_sharded_operand_and_indices,
      @parallel_scatter_min_ui8_sharded_operand_and_indices
    ]]
  } : (tensor<6x2xi32>, tensor<5x1xui8>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xui8>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xui8>, tensor<5x2xi32>,
       tensor<6x2xi32>, tensor<5x1xui8>, tensor<5x2xi32>)
      -> (tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>)
  "check.expect_eq"(%res_ui8#0, %seq_ui8) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_ui8#1, %seq_ui8) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_ui8#2, %seq_ui8) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%res_ui8#3, %seq_ui8) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()

  %in300 = stablehlo.constant dense<0> : tensor<300xi32>
  %idx3_ui8 = stablehlo.constant dense<[[10], [20], [30]]> : tensor<3x1xui8>
  %upd3 = stablehlo.constant dense<[100, 200, 300]> : tensor<3xi32>
  %seq_ui8_300 = func.call @sequential_scatter_indivisible_ui8_indices_large_operand(%in300, %idx3_ui8, %upd3)
      : (tensor<300xi32>, tensor<3x1xui8>, tensor<3xi32>) -> tensor<300xi32>
  %res_ui8_300:4 = "interpreter.run_parallel"(
      %in300, %idx3_ui8, %upd3,
      %in300, %idx3_ui8, %upd3,
      %in300, %idx3_ui8, %upd3,
      %in300, %idx3_ui8, %upd3) {
    programs = [[
      @parallel_scatter_indivisible_ui8_indices_large_operand,
      @parallel_scatter_indivisible_ui8_indices_large_operand,
      @parallel_scatter_indivisible_ui8_indices_large_operand,
      @parallel_scatter_indivisible_ui8_indices_large_operand
    ]]
  } : (tensor<300xi32>, tensor<3x1xui8>, tensor<3xi32>,
       tensor<300xi32>, tensor<3x1xui8>, tensor<3xi32>,
       tensor<300xi32>, tensor<3x1xui8>, tensor<3xi32>,
       tensor<300xi32>, tensor<3x1xui8>, tensor<3xi32>)
      -> (tensor<300xi32>, tensor<300xi32>, tensor<300xi32>, tensor<300xi32>)
  "check.expect_eq"(%res_ui8_300#0, %seq_ui8_300) : (tensor<300xi32>, tensor<300xi32>) -> ()
  "check.expect_eq"(%res_ui8_300#1, %seq_ui8_300) : (tensor<300xi32>, tensor<300xi32>) -> ()
  "check.expect_eq"(%res_ui8_300#2, %seq_ui8_300) : (tensor<300xi32>, tensor<300xi32>) -> ()
  "check.expect_eq"(%res_ui8_300#3, %seq_ui8_300) : (tensor<300xi32>, tensor<300xi32>) -> ()

  %in150 = stablehlo.constant dense<0> : tensor<150xi32>
  %upd1 = stablehlo.constant dense<[999]> : tensor<1xi32>
  %idx_i8_dus = stablehlo.constant dense<-100> : tensor<i8>
  %seq_dus_i8 = func.call @sequential_dynamic_update_slice_i8_large_operand(%in150, %upd1, %idx_i8_dus)
      : (tensor<150xi32>, tensor<1xi32>, tensor<i8>) -> tensor<150xi32>
  %res_dus_i8:4 = "interpreter.run_parallel"(
      %in150, %upd1, %idx_i8_dus,
      %in150, %upd1, %idx_i8_dus,
      %in150, %upd1, %idx_i8_dus,
      %in150, %upd1, %idx_i8_dus) {
    programs = [[
      @parallel_dynamic_update_slice_i8_large_operand,
      @parallel_dynamic_update_slice_i8_large_operand,
      @parallel_dynamic_update_slice_i8_large_operand,
      @parallel_dynamic_update_slice_i8_large_operand
    ]]
  } : (tensor<150xi32>, tensor<1xi32>, tensor<i8>,
       tensor<150xi32>, tensor<1xi32>, tensor<i8>,
       tensor<150xi32>, tensor<1xi32>, tensor<i8>,
       tensor<150xi32>, tensor<1xi32>, tensor<i8>)
      -> (tensor<150xi32>, tensor<150xi32>, tensor<150xi32>, tensor<150xi32>)
  "check.expect_eq"(%res_dus_i8#0, %seq_dus_i8) : (tensor<150xi32>, tensor<150xi32>) -> ()
  "check.expect_eq"(%res_dus_i8#1, %seq_dus_i8) : (tensor<150xi32>, tensor<150xi32>) -> ()
  "check.expect_eq"(%res_dus_i8#2, %seq_dus_i8) : (tensor<150xi32>, tensor<150xi32>) -> ()
  "check.expect_eq"(%res_dus_i8#3, %seq_dus_i8) : (tensor<150xi32>, tensor<150xi32>) -> ()
  return
}
