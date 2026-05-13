// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_concat(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
  %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1
  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
  : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  %c0 = stablehlo.constant dense<[
    [1, 2], [3, 4],
    [5, 6], [7, 8]
  ]> : tensor<4x2xi32>
  %c1 = stablehlo.constant dense<[
    [10, 20], [30, 40],
    [50, 60], [70, 80]
  ]> : tensor<4x2xi32>

  %seq = func.call @sequential_concat(%c0, %c1) : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x4xi32>

  // Manually prepare shards for 2 devices (sharded along dimension 0).
  // Device 0: Rows [0:2]
  %s0_0 = "stablehlo.slice"(%c0) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %s1_0 = "stablehlo.slice"(%c1) {start_indices=array<i64: 0,0>, limit_indices=array<i64: 2,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  // Device 1: Rows [2:4]
  %s0_1 = "stablehlo.slice"(%c0) {start_indices=array<i64: 2,0>, limit_indices=array<i64: 4,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %s1_1 = "stablehlo.slice"(%c1) {start_indices=array<i64: 2,0>, limit_indices=array<i64: 4,2>, strides=array<i64: 1,1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %res:2 = "interpreter.run_parallel"(%s0_0, %s1_0, %s0_1, %s1_1) {
    programs = [[@parallel_concat, @parallel_concat]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) ->
      (tensor<2x4xi32>, tensor<2x4xi32>)

  %par = "stablehlo.concatenate"(%res#0, %res#1) {dimension = 0 : i64} : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<4x4xi32>
  "check.expect_eq"(%par, %seq) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()

  return
}
