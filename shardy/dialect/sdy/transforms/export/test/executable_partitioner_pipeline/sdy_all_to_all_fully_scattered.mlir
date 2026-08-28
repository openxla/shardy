// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"

// Resharding pattern:
// - Mesh: 8 devices @mesh = <["z"=2, "x"=2, "y"=2]>.
// - Resharding: [{"z", "x", "y"}, {}, {}] -> [{"z"}, {"x"}, {"y"}].
//
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
// dim 0 while leaving "z" untouched) followed by a chain of two sdy.all_to_all
// ops (moving "x" to dim 1 and "y" to dim 2).
// OptimizeCollectivesPass eliminates the collective_permute by decomposing
// dim 0 shape and executing the combined all_to_all, keeping "z" untouched on
// dim 0.
// The permuted axes ("x" and "y") are fully scattered off dimension 0 to
// dimensions 1 and 2, while the unpermuted axis "z" remains untouched on
// dimension 0.
//
// Numerically verified via interpreter.run_parallel.

//--- part1.mlir
sdy.mesh @mesh = <["z"=2, "x"=2, "y"=2]>

func.func @parallel_sdy_all_to_all_fully_scattered(%arg0: tensor<8x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z", "x", "y"}, {}, {}]>}) -> (tensor<8x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"x"}, {"y"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{"z"}, {"x"}, {"y"}]> : tensor<8x2x2xf32>
  return %0 : tensor<8x2x2xf32>
}

func.func @sequential_sdy_all_to_all_fully_scattered(%arg0: tensor<8x2x2xf32>) -> tensor<8x2x2xf32> {
  return %arg0 : tensor<8x2x2xf32>
}

//--- part2.mlir
func.func @main() {
  %input = stablehlo.constant dense<[
    [[ 0.0,  1.0], [ 2.0,  3.0]],
    [[ 4.0,  5.0], [ 6.0,  7.0]],
    [[ 8.0,  9.0], [10.0, 11.0]],
    [[12.0, 13.0], [14.0, 15.0]],
    [[16.0, 17.0], [18.0, 19.0]],
    [[20.0, 21.0], [22.0, 23.0]],
    [[24.0, 25.0], [26.0, 27.0]],
    [[28.0, 29.0], [30.0, 31.0]]
  ]> : tensor<8x2x2xf32>
  %seq = func.call @sequential_sdy_all_to_all_fully_scattered(%input) : (tensor<8x2x2xf32>) -> tensor<8x2x2xf32>

  // Input slices (dim 0 sharded by {"z", "x", "y"} across 8 devices):
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 1, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 1, 0, 0>, limit_indices = array<i64: 2, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0, 0>, limit_indices = array<i64: 3, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3, 0, 0>, limit_indices = array<i64: 4, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>
  %s4 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0, 0>, limit_indices = array<i64: 5, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>
  %s5 = "stablehlo.slice"(%input) {start_indices = array<i64: 5, 0, 0>, limit_indices = array<i64: 6, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>
  %s6 = "stablehlo.slice"(%input) {start_indices = array<i64: 6, 0, 0>, limit_indices = array<i64: 7, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>
  %s7 = "stablehlo.slice"(%input) {start_indices = array<i64: 7, 0, 0>, limit_indices = array<i64: 8, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<1x2x2xf32>

  %res:8 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7) {
    programs = [[
      @parallel_sdy_all_to_all_fully_scattered, @parallel_sdy_all_to_all_fully_scattered,
      @parallel_sdy_all_to_all_fully_scattered, @parallel_sdy_all_to_all_fully_scattered,
      @parallel_sdy_all_to_all_fully_scattered, @parallel_sdy_all_to_all_fully_scattered,
      @parallel_sdy_all_to_all_fully_scattered, @parallel_sdy_all_to_all_fully_scattered
    ]]
  } : (tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>,
       tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>) ->
      (tensor<4x1x1xf32>, tensor<4x1x1xf32>, tensor<4x1x1xf32>, tensor<4x1x1xf32>,
       tensor<4x1x1xf32>, tensor<4x1x1xf32>, tensor<4x1x1xf32>, tensor<4x1x1xf32>)

  // Expected output slices (dim 0 sharded by "z", dim 1 by "x", dim 2 by "y"):
  // Device 0 (z=0, x=0, y=0)
  %exp0 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 4, 1, 1>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>
  // Device 1 (z=0, x=0, y=1)
  %exp1 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 0, 1>, limit_indices = array<i64: 4, 1, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>
  // Device 2 (z=0, x=1, y=0)
  %exp2 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 1, 0>, limit_indices = array<i64: 4, 2, 1>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>
  // Device 3 (z=0, x=1, y=1)
  %exp3 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 1, 1>, limit_indices = array<i64: 4, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>
  // Device 4 (z=1, x=0, y=0)
  %exp4 = "stablehlo.slice"(%seq) {start_indices = array<i64: 4, 0, 0>, limit_indices = array<i64: 8, 1, 1>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>
  // Device 5 (z=1, x=0, y=1)
  %exp5 = "stablehlo.slice"(%seq) {start_indices = array<i64: 4, 0, 1>, limit_indices = array<i64: 8, 1, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>
  // Device 6 (z=1, x=1, y=0)
  %exp6 = "stablehlo.slice"(%seq) {start_indices = array<i64: 4, 1, 0>, limit_indices = array<i64: 8, 2, 1>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>
  // Device 7 (z=1, x=1, y=1)
  %exp7 = "stablehlo.slice"(%seq) {start_indices = array<i64: 4, 1, 1>, limit_indices = array<i64: 8, 2, 2>, strides = array<i64: 1, 1, 1>} : (tensor<8x2x2xf32>) -> tensor<4x1x1xf32>

  "check.expect_eq"(%res#0, %exp0) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  "check.expect_eq"(%res#1, %exp1) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  "check.expect_eq"(%res#2, %exp2) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  "check.expect_eq"(%res#3, %exp3) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  "check.expect_eq"(%res#4, %exp4) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  "check.expect_eq"(%res#5, %exp5) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  "check.expect_eq"(%res#6, %exp6) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  "check.expect_eq"(%res#7, %exp7) : (tensor<4x1x1xf32>, tensor<4x1x1xf32>) -> ()
  return
}
