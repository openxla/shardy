// RUN: split-file %s %t

// Convert parallel_gather to a device function.
// RUN: sdy_opt %t/part1.mlir --sdy-convert-global-to-local --sdy-drop-sharding-and-mesh \
// RUN:         --allow-unregistered-dialect > %t/part1_processed.mlir

// Assemble the final module.
// RUN: sed '1d; /^}/,$d' %t/part1_processed.mlir > %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir

// Execute via the interpreter.
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// Sequential Baseline: Explicitly written, no shardings.
func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// Parallel in global view, to be converted to a device function.
//
// ([i, j], [k]) -> ([k, j]) reduction={i}
// The sharded dim is a reduction dimensions and it is also a collapsed dim.
func.func @parallel_gather(
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 2>
  } : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
  %indices = stablehlo.constant dense<[1, 3]> : tensor<2xi64>

  %seq = func.call @sequential_gather(%input, %indices) : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>

  %shard0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %shard1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars:2 = "interpreter.run_parallel"(%shard0, %indices, %shard1, %indices) {
    programs = [[@parallel_gather, @parallel_gather]]
  } : (tensor<2x2xf32>, tensor<2xi64>, tensor<2x2xf32>, tensor<2xi64>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  return
}
