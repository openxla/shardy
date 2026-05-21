// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_select_and_scatter(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
                                       %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %init) <{
    window_dimensions = array<i64: 1, 2>,
    window_strides = array<i64: 1, 2>,
    padding = dense<0> : tensor<2x2xi64>
  }> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %cmp = stablehlo.compare GE, %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }, {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %add = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<4x4xf32>, tensor<4x2xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

//--- part2.mlir

func.func @main() {
  %operand = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                                      [100.0, 200.0, 300.0, 400.0], [500.0, 600.0, 700.0, 800.0]]> : tensor<4x4xf32>
  %source = stablehlo.constant dense<[[10.0, 20.0], [30.0, 40.0], [10.0, 20.0], [30.0, 40.0]]> : tensor<4x2xf32>

  %seq = func.call @sequential_select_and_scatter(%operand, %source) : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>

  %operand_shard0 = "stablehlo.slice"(%operand) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %operand_shard1 = "stablehlo.slice"(%operand) {
    start_indices = array<i64: 2, 0>,
    limit_indices = array<i64: 4, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<4x4xf32>) -> tensor<2x4xf32>

  %source_shard0 = "stablehlo.slice"(%source) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 2, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %source_shard1 = "stablehlo.slice"(%source) {
    start_indices = array<i64: 2, 0>,
    limit_indices = array<i64: 4, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<4x2xf32>) -> tensor<2x2xf32>

  %pars:2 = "interpreter.run_parallel"(%operand_shard0, %source_shard0, %operand_shard1, %source_shard1) {
    programs = [[@parallel_select_and_scatter, @parallel_select_and_scatter]]
  } : (tensor<2x4xf32>, tensor<2x2xf32>, tensor<2x4xf32>, tensor<2x2xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {
    dimension = 0 : i64
  } : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>

  check.expect_eq %seq, %par : tensor<4x4xf32>

  return
}
