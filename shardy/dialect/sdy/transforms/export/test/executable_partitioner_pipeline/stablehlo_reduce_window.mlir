// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_reduce_window_max(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    window_dimensions = array<i64: 1, 2>,
    window_strides = array<i64: 1, 2>
  }> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @parallel_reduce_window_add(
    %arg0: tensor<4x6xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    padding = dense<[[0, 0], [1, 1]]> : tensor<2x2xi64>,
    window_dimensions = array<i64: 1, 3>,
    window_strides = array<i64: 1, 2>
  }> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

func.func @parallel_reduce_window_base_dilation(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{
    base_dilations = array<i64: 3, 1>,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  }> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<2x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

//--- part2.mlir

func.func @main() {
  %input_max = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>

  %seq_max = func.call @sequential_reduce_window_max(%input_max) : (tensor<2x4xf32>) -> tensor<2x2xf32>

  %shard0_max = "stablehlo.slice"(%input_max) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 1, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<2x4xf32>) -> tensor<1x4xf32>
  %shard1_max = "stablehlo.slice"(%input_max) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<2x4xf32>) -> tensor<1x4xf32>

  %pars_max:2 = "interpreter.run_parallel"(%shard0_max, %shard1_max) {
    programs = [[@parallel_reduce_window_max, @parallel_reduce_window_max]]
  } : (tensor<1x4xf32>, tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)

  %par_max = "stablehlo.concatenate"(%pars_max#0, %pars_max#1) {
    dimension = 0 : i64
  } : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>

  check.expect_eq %seq_max, %par_max : tensor<2x2xf32>

  %input_add = stablehlo.constant dense<[
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
  ]> : tensor<4x6xf32>

  %seq_add = func.call @sequential_reduce_window_add(%input_add) : (tensor<4x6xf32>) -> tensor<4x3xf32>

  %shard0_add = "stablehlo.slice"(%input_add) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 6>,
    strides = array<i64: 1, 1>
  } : (tensor<4x6xf32>) -> tensor<2x6xf32>
  %shard1_add = "stablehlo.slice"(%input_add) {
    start_indices = array<i64: 2, 0>,
    limit_indices = array<i64: 4, 6>,
    strides = array<i64: 1, 1>
  } : (tensor<4x6xf32>) -> tensor<2x6xf32>

  %pars_add:2 = "interpreter.run_parallel"(%shard0_add, %shard1_add) {
    programs = [[@parallel_reduce_window_add, @parallel_reduce_window_add]]
  } : (tensor<2x6xf32>, tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>)

  %par_add = "stablehlo.concatenate"(%pars_add#0, %pars_add#1) {
    dimension = 0 : i64
  } : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>

  check.expect_eq %seq_add, %par_add : tensor<4x3xf32>

  %seq_base_dil = func.call @sequential_reduce_window_base_dilation(%input_max) : (tensor<2x4xf32>) -> tensor<4x4xf32>

  %pars_base_dil:2 = "interpreter.run_parallel"(%shard0_max, %shard1_max) {
    programs = [[@parallel_reduce_window_base_dilation, @parallel_reduce_window_base_dilation]]
  } : (tensor<1x4xf32>, tensor<1x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

  %par_base_dil = "stablehlo.concatenate"(%pars_base_dil#0, %pars_base_dil#1) {
    dimension = 0 : i64
  } : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>

  check.expect_eq %seq_base_dil, %par_base_dil : tensor<4x4xf32>

  return
}
