// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"

// Resharding pattern:
// - Mesh: 4 devices @mesh = <["x"=2, "y"=2]>.
// - Resharding: [{"x", "y"}, {}] -> [{"y"}, {"x"}].
//
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
// dim 0) followed by sdy.all_to_all (scattering "x" to dim 1).
// OptimizeCollectivesPass detects the collective_permute + all_to_all chain and
// eliminates the collective_permute by decomposing dim 0 shape and issuing a
// single multi-axis all_to_all.
// The permuted axes ("x" and "y") are partially scattered off dimension 0 (only
// "x" is scattered to dimension 1, while "y" remains on dimension 0).
//
// Numerically verified via interpreter.run_parallel.

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @parallel_sdy_all_to_all_partially_scattered(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {"x"}]> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @sequential_sdy_all_to_all_partially_scattered(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  return %arg0 : tensor<4x4xf32>
}

//--- part2.mlir
func.func @main() {
  %input = stablehlo.constant dense<[
    [ 0.0,  1.0,  2.0,  3.0],
    [ 4.0,  5.0,  6.0,  7.0],
    [ 8.0,  9.0, 10.0, 11.0],
    [12.0, 13.0, 14.0, 15.0]
  ]> : tensor<4x4xf32>
  %seq = func.call @sequential_sdy_all_to_all_partially_scattered(%input) : (tensor<4x4xf32>) -> tensor<4x4xf32>

  // Input slices (dim 0 sharded by {"x", "y"} across 4 devices):
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[
      @parallel_sdy_all_to_all_partially_scattered, @parallel_sdy_all_to_all_partially_scattered,
      @parallel_sdy_all_to_all_partially_scattered, @parallel_sdy_all_to_all_partially_scattered
    ]]
  } : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) ->
      (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  // Expected output slices (dim 0 sharded by "y", dim 1 sharded by "x"):
  // Device 0 (x=0, y=0)
  %exp0 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  // Device 1 (x=0, y=1)
  %exp1 = "stablehlo.slice"(%seq) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  // Device 2 (x=1, y=0)
  %exp2 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  // Device 3 (x=1, y=1)
  %exp3 = "stablehlo.slice"(%seq) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>

  "check.expect_eq"(%res#0, %exp0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%res#1, %exp1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%res#2, %exp2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%res#3, %exp3) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  return
}
