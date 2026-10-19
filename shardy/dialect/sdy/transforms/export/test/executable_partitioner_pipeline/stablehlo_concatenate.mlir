// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

func.func @parallel_concat(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>},
  %arg1: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>})
  -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {"x"}]>]>}
    : (tensor<4x2xi32>, tensor<4x4xi32>) -> tensor<4x6xi32>
  return %0 : tensor<4x6xi32>
}

//--- part2.mlir

func.func @main() {
  %c0 = stablehlo.constant dense<[
    [1, 2], [3, 4],
    [5, 6], [7, 8]
  ]> : tensor<4x2xi32>
  %c1 = stablehlo.constant dense<[
    [10, 20, 30, 40], [50, 60, 70, 80],
    [90, 100, 110, 120], [130, 140, 150, 160]
  ]> : tensor<4x4xi32>

  %seq = func.call @sequential_concat(%c0, %c1) : (tensor<4x2xi32>, tensor<4x4xi32>) -> tensor<4x6xi32>

  // Manually prepare shards for 2 devices (sharded along dimension 1).
  // Device 0: Columns [0:1] of %c0 and [0:2] of %c1
  %s0_0 = "stablehlo.slice"(%c0) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 1>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<4x1xi32>
  %s1_0 = "stablehlo.slice"(%c1) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x2xi32>

  // Device 1: Columns [1:2] of %c0 and [2:4] of %c1
  %s0_1 = "stablehlo.slice"(%c0) {start_indices = array<i64: 0, 1>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<4x1xi32>
  %s1_1 = "stablehlo.slice"(%c1) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x2xi32>

  %res:2 = "interpreter.run_parallel"(%s0_0, %s1_0, %s0_1, %s1_1) {
    programs = [[@parallel_concat, @parallel_concat]]
  } : (tensor<4x1xi32>, tensor<4x2xi32>, tensor<4x1xi32>, tensor<4x2xi32>) ->
      (tensor<4x6xi32>, tensor<4x6xi32>)

  check.expect_eq %seq, %res#0 : tensor<4x6xi32>
  check.expect_eq %seq, %res#1 : tensor<4x6xi32>

  return
}
