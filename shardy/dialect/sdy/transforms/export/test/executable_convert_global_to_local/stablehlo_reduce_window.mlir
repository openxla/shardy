// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_reduce_window(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
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

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>

  %seq = func.call @sequential_reduce_window(%input) : (tensor<2x4xf32>) -> tensor<2x2xf32>

  %shard0 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 1, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<2x4xf32>) -> tensor<1x4xf32>
  %shard1 = "stablehlo.slice"(%input) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<2x4xf32>) -> tensor<1x4xf32>

  %pars:2 = "interpreter.run_parallel"(%shard0, %shard1) {
    programs = [[@parallel_reduce_window, @parallel_reduce_window]]
  } : (tensor<1x4xf32>, tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)

  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {
    dimension = 0 : i64
  } : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>

  check.expect_eq %seq, %par : tensor<2x2xf32>

  return
}
