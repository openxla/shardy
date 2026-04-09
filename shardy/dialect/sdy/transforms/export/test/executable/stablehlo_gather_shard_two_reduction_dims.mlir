// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

func.func @sequential_gather(%arg0: tensor<4x2x2xf32>, %arg1: tensor<2x1x2xi64>) -> tensor<2x1xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0, 2],
      operand_batching_dims = [1],
      start_indices_batching_dims = [0],
      start_index_map = [0, 2],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 1>
  } : (tensor<4x2x2xf32>, tensor<2x1x2xi64>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// ([r1, b, r2], [b, i, v]) -> ([b, i]) reduction={r1, r2}
//
func.func @parallel_gather(
    %arg0: tensor<4x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}, {"y"}]>},
    %arg1: tensor<2x1x2xi64>) -> tensor<2x1xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0, 2],
      operand_batching_dims = [1],
      start_indices_batching_dims = [0],
      start_index_map = [0, 2],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 1>
  } : (tensor<4x2x2xf32>, tensor<2x1x2xi64>) -> tensor<2x1xf32>

  // All-reduce across both sharded reduction axes.
  %1 = sdy.all_reduce {"x", "y"} %0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

//--- part2.mlir

func.func @main() {
  // Input: tensor<4x2x2xf32>
  %input = stablehlo.constant dense<[
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]],
    [[9.0, 10.0], [11.0, 12.0]],
    [[13.0, 14.0], [15.0, 16.0]]
  ]> : tensor<4x2x2xf32>
  %indices = stablehlo.constant dense<[[[1, 1]], [[3, 0]]]> : tensor<2x1x2xi64>

  %seq = func.call @sequential_gather(%input, %indices) : (tensor<4x2x2xf32>, tensor<2x1x2xi64>) -> tensor<2x1xf32>

  // Prepare Shards (4 devices: x=2, y=2)
  // Shard 0 (x=0, y=0): r1:[0,1], b:[0,1], r2:0
  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0,0,0>, limit_indices=array<i64: 2,2,1>, strides=array<i64: 1,1,1>} : (tensor<4x2x2xf32>) -> tensor<2x2x1xf32>
  // Shard 1 (x=0, y=1): r1:[0,1], b:[0,1], r2:1
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 0,0,1>, limit_indices=array<i64: 2,2,2>, strides=array<i64: 1,1,1>} : (tensor<4x2x2xf32>) -> tensor<2x2x1xf32>
  // Shard 2 (x=1, y=0): r1:[2,3], b:[0,1], r2:0
  %s2 = "stablehlo.slice"(%input) {start_indices=array<i64: 2,0,0>, limit_indices=array<i64: 4,2,1>, strides=array<i64: 1,1,1>} : (tensor<4x2x2xf32>) -> tensor<2x2x1xf32>
  // Shard 3 (x=1, y=1): r1:[2,3], b:[0,1], r2:1
  %s3 = "stablehlo.slice"(%input) {start_indices=array<i64: 2,0,1>, limit_indices=array<i64: 4,2,2>, strides=array<i64: 1,1,1>} : (tensor<4x2x2xf32>) -> tensor<2x2x1xf32>

  %pars:4 = "interpreter.run_parallel"(%s0, %indices, %s1, %indices, %s2, %indices, %s3, %indices) {
    programs = [[@parallel_gather, @parallel_gather, @parallel_gather, @parallel_gather]]
  } : (tensor<2x2x1xf32>, tensor<2x1x2xi64>, tensor<2x2x1xf32>, tensor<2x1x2xi64>,
       tensor<2x2x1xf32>, tensor<2x1x2xi64>, tensor<2x2x1xf32>, tensor<2x1x2xi64>) ->
      (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>)

  "check.expect_eq"(%pars#0, %seq) : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
  "check.expect_eq"(%pars#1, %seq) : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
  "check.expect_eq"(%pars#2, %seq) : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
  "check.expect_eq"(%pars#3, %seq) : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
  return
}
