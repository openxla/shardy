// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// 1. Gather with indivisible operand padding along sliced/indexed dimension
// with implicit 1-element index vector (index_vector_dim == rank).
// The operand is sliced from 4x2 to 3x2, sharded on {"x"} along dim 0.
// Mesh size is 2, so dim 0 (size 3) is indivisible and padded to 4.
// Start indices are clamped to 3 - 1 = 2 for in-bounds and out-of-bounds start indices.
func.func @parallel_gather_indivisible_padding(
    %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<3x1xi32>)
    -> (tensor<3x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %operand = stablehlo.slice %arg0 [0:3, 0:2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
    : (tensor<4x2xf32>) -> tensor<3x2xf32>
  %0 = "stablehlo.gather"(%operand, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {}]>]>
  } : (tensor<3x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @sequential_gather_indivisible_padding(
    %arg0: tensor<4x2xf32>,
    %arg1: tensor<3x1xi32>) -> tensor<3x2xf32> {
  %operand = stablehlo.slice %arg0 [0:3, 0:2] : (tensor<4x2xf32>) -> tensor<3x2xf32>
  %0 = "stablehlo.gather"(%operand, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<3x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// 2. Gather with explicit index vector (index_vector_dim < rank) and multi-dim indexing.
// The operand is sliced from 4x4 to 3x3, sharded on {"x"} along dim 0.
// Start indices are 2x2 with start_index_map = [0, 1].
// Clamping bounds [2, 2] are broadcast across index_vector_dim to clamp OOB indices.
func.func @parallel_gather_explicit_index_vector(
    %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
    %arg1: tensor<2x2xi32>)
    -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}]>}) {
  %operand = stablehlo.slice %arg0 [0:3, 0:3]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
    : (tensor<4x4xf32>) -> tensor<3x3xf32>
  %0 = "stablehlo.gather"(%operand, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}]>]>
  } : (tensor<3x3xf32>, tensor<2x2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @sequential_gather_explicit_index_vector(
    %arg0: tensor<4x4xf32>,
    %arg1: tensor<2x2xi32>) -> tensor<2xf32> {
  %operand = stablehlo.slice %arg0 [0:3, 0:3] : (tensor<4x4xf32>) -> tensor<3x3xf32>
  %0 = "stablehlo.gather"(%operand, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>
  } : (tensor<3x3xf32>, tensor<2x2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

//--- part2.mlir

func.func @main() {
  // Test 1: Implicit 1-element index vector.
  %input = stablehlo.constant dense<[
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
  ]> : tensor<4x2xf32>

  // Index 0 (valid), Index 2 (valid boundary), Index 5 (out-of-bounds -> clamped to 2).
  %indices = stablehlo.constant dense<[[0], [2], [5]]> : tensor<3x1xi32>

  %seq = func.call @sequential_gather_indivisible_padding(%input, %indices)
    : (tensor<4x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>

  // Slice input into 2 shards of size 2x2.
  %s0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %s1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %res:2 = "interpreter.run_parallel"(%s0, %indices, %s1, %indices) {
    programs = [[@parallel_gather_indivisible_padding, @parallel_gather_indivisible_padding]]
  } : (tensor<2x2xf32>, tensor<3x1xi32>, tensor<2x2xf32>, tensor<3x1xi32>)
    -> (tensor<3x2xf32>, tensor<3x2xf32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()

  // Test 2: Explicit index vector with 2D coordinates.
  %input_2d = stablehlo.constant dense<[
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
  ]> : tensor<4x4xf32>

  // Point [0, 1] (valid), Point [4, 1] (row 4 out-of-bounds -> clamped to row 2).
  %indices_2d = stablehlo.constant dense<[
    [0, 1],
    [4, 1]
  ]> : tensor<2x2xi32>

  %seq_2d = func.call @sequential_gather_explicit_index_vector(%input_2d, %indices_2d)
    : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<2xf32>

  // Slice input into 2 shards of size 2x4.
  %s0_2d = "stablehlo.slice"(%input_2d) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>
  } : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %s1_2d = "stablehlo.slice"(%input_2d) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>
  } : (tensor<4x4xf32>) -> tensor<2x4xf32>

  %res_2d:2 = "interpreter.run_parallel"(%s0_2d, %indices_2d, %s1_2d, %indices_2d) {
    programs = [[@parallel_gather_explicit_index_vector, @parallel_gather_explicit_index_vector]]
  } : (tensor<2x4xf32>, tensor<2x2xi32>, tensor<2x4xf32>, tensor<2x2xi32>)
    -> (tensor<2xf32>, tensor<2xf32>)

  "check.expect_eq"(%res_2d#0, %seq_2d) : (tensor<2xf32>, tensor<2xf32>) -> ()
  "check.expect_eq"(%res_2d#1, %seq_2d) : (tensor<2xf32>, tensor<2xf32>) -> ()

  return
}
