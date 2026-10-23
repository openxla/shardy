// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// Test 1: the sort dimension itself is sharded, so the input is resharded
// before the sort and back after it.
func.func @parallel_sort_sharded_sort_dim(
    %arg0: tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %p = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%p) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = true,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<4x6xi32>) -> tensor<4x6xi32>
  return %0 : tensor<4x6xi32>
}

// Test 2: 2D mesh with batch dimensions 0 and 1 sharded and sort along dimension 2.
func.func @parallel_sort_2d_mesh(
    %arg0: tensor<4x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}, {}]>})
    -> (tensor<4x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}, {}]>}) {
  %0 = "stablehlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %p = "stablehlo.compare"(%arg1, %arg2) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%p) : (tensor<i1>) -> ()
  }) {dimension = 2 : i64, is_stable = true,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}, {}]>]>
  } : (tensor<4x4x4xi32>) -> tensor<4x4x4xi32>
  return %0 : tensor<4x4x4xi32>
}

//--- part2.mlir

func.func @main() {
  // Test 1: sort dimension sharded. Every column has values that must move
  // between the two shards.
  %input_1d = stablehlo.constant dense<[
    [40, 11, 32, 23, 44, 15], [20, 31, 12, 43, 24, 35],
    [10, 41, 22, 13, 34, 45], [30, 21, 42, 33, 14, 25]
  ]> : tensor<4x6xi32>

  %s0_1d = "stablehlo.slice"(%input_1d) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 2, 6>, strides=array<i64: 1, 1>} : (tensor<4x6xi32>) -> tensor<2x6xi32>
  %s1_1d = "stablehlo.slice"(%input_1d) {start_indices=array<i64: 2, 0>, limit_indices=array<i64: 4, 6>, strides=array<i64: 1, 1>} : (tensor<4x6xi32>) -> tensor<2x6xi32>

  %seq_1d = func.call @sequential_sort_sharded_sort_dim(%input_1d) : (tensor<4x6xi32>) -> tensor<4x6xi32>

  %pars_1d:2 = "interpreter.run_parallel"(%s0_1d, %s1_1d) {
    programs = [[@parallel_sort_sharded_sort_dim, @parallel_sort_sharded_sort_dim]]
  } : (tensor<2x6xi32>, tensor<2x6xi32>) -> (tensor<2x6xi32>, tensor<2x6xi32>)
  %par_1d = "stablehlo.concatenate"(%pars_1d#0, %pars_1d#1) {dimension = 0 : i64}
    : (tensor<2x6xi32>, tensor<2x6xi32>) -> tensor<4x6xi32>

  check.expect_eq %seq_1d, %par_1d : tensor<4x6xi32>

  // Test 2: 2D mesh with batch dimensions 0 and 1 sharded.
  %input_2d = stablehlo.constant dense<[
    [[4, 1, 3, 2], [8, 5, 7, 6], [12, 9, 11, 10], [16, 13, 15, 14]],
    [[3, 2, 4, 1], [7, 6, 8, 5], [11, 10, 12, 9], [15, 14, 16, 13]],
    [[40, 10, 30, 20], [80, 50, 70, 60], [120, 90, 110, 100], [160, 130, 150, 140]],
    [[30, 20, 40, 10], [70, 60, 80, 50], [110, 100, 120, 90], [150, 140, 160, 130]]
  ]> : tensor<4x4x4xi32>

  %s00 = "stablehlo.slice"(%input_2d) {start_indices=array<i64: 0, 0, 0>, limit_indices=array<i64: 2, 2, 4>, strides=array<i64: 1, 1, 1>} : (tensor<4x4x4xi32>) -> tensor<2x2x4xi32>
  %s01 = "stablehlo.slice"(%input_2d) {start_indices=array<i64: 0, 2, 0>, limit_indices=array<i64: 2, 4, 4>, strides=array<i64: 1, 1, 1>} : (tensor<4x4x4xi32>) -> tensor<2x2x4xi32>
  %s10 = "stablehlo.slice"(%input_2d) {start_indices=array<i64: 2, 0, 0>, limit_indices=array<i64: 4, 2, 4>, strides=array<i64: 1, 1, 1>} : (tensor<4x4x4xi32>) -> tensor<2x2x4xi32>
  %s11 = "stablehlo.slice"(%input_2d) {start_indices=array<i64: 2, 2, 0>, limit_indices=array<i64: 4, 4, 4>, strides=array<i64: 1, 1, 1>} : (tensor<4x4x4xi32>) -> tensor<2x2x4xi32>

  %seq_2d = func.call @sequential_sort_2d_mesh(%input_2d) : (tensor<4x4x4xi32>) -> tensor<4x4x4xi32>

  %pars_2d:4 = "interpreter.run_parallel"(%s00, %s01, %s10, %s11) {
    programs = [[@parallel_sort_2d_mesh, @parallel_sort_2d_mesh, @parallel_sort_2d_mesh, @parallel_sort_2d_mesh]]
  } : (tensor<2x2x4xi32>, tensor<2x2x4xi32>, tensor<2x2x4xi32>, tensor<2x2x4xi32>) -> (tensor<2x2x4xi32>, tensor<2x2x4xi32>, tensor<2x2x4xi32>, tensor<2x2x4xi32>)

  %row01 = "stablehlo.concatenate"(%pars_2d#0, %pars_2d#1) {dimension = 1 : i64} : (tensor<2x2x4xi32>, tensor<2x2x4xi32>) -> tensor<2x4x4xi32>
  %row23 = "stablehlo.concatenate"(%pars_2d#2, %pars_2d#3) {dimension = 1 : i64} : (tensor<2x2x4xi32>, tensor<2x2x4xi32>) -> tensor<2x4x4xi32>
  %par_2d = "stablehlo.concatenate"(%row01, %row23) {dimension = 0 : i64} : (tensor<2x4x4xi32>, tensor<2x4x4xi32>) -> tensor<4x4x4xi32>

  check.expect_eq %seq_2d, %par_2d : tensor<4x4x4xi32>

  return
}
