// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @sequential_gather(%arg0: tensor<4x2xf32>, %arg1: tensor<2xi64>) -> tensor<2x1xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>
  } : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// ([i, j], [k]) -> ([k, j])  reduction={i}
//
func.func @parallel_gather(
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<2xi64>) -> tensor<2x1xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>
  } : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x1xf32>

  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2, [{}, {}]> : tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
  %indices = stablehlo.constant dense<[0, 1]> : tensor<2xi64>

  %seq = func.call @sequential_gather(%input, %indices) : (tensor<4x2xf32>, tensor<2xi64>) -> tensor<2x1xf32>

  %shard0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %shard1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars:2 = "interpreter.run_parallel"(%shard0, %indices, %shard1, %indices) {
    programs = [[@parallel_gather, @parallel_gather]]
  } : (tensor<2x2xf32>, tensor<2xi64>, tensor<2x2xf32>, tensor<2xi64>) -> (tensor<2x1xf32>, tensor<2x1xf32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
  return
}
