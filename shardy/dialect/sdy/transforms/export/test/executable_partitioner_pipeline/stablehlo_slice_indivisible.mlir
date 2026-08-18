// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh_x4 = <["x"=4]>

// The slice result size (3) along the sharded dimension is not divisible by the
// mesh axis size (4). PadForDivisibility should pad the slice result to 4,
// and then trim it back to 3 after the reshard (which turns into all_gather).
func.func @parallel_slice_result_indivisible(
  %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{"x"}, {}]>})
  -> (tensor<3x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:3, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{"x"}, {}]>]>} : (tensor<4x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

func.func @sequential_slice_result_indivisible(%arg0: tensor<4x2xf32>) -> tensor<3x2xf32> {
  %0 = stablehlo.slice %arg0 [0:3, 0:2] : (tensor<4x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[
    [1.02, -2.45],
    [4.03, 5.76],
    [-7.34, 8.91],
    [0.12, -9.54]
  ]> : tensor<4x2xf32>

  %seq = func.call @sequential_slice_result_indivisible(%input) : (tensor<4x2xf32>) -> tensor<3x2xf32>

  // Slice input into 4 shards of size 1x2.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xf32>) -> tensor<1x2xf32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xf32>) -> tensor<1x2xf32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xf32>) -> tensor<1x2xf32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xf32>) -> tensor<1x2xf32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_slice_result_indivisible, @parallel_slice_result_indivisible, @parallel_slice_result_indivisible, @parallel_slice_result_indivisible]]
  } : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>, tensor<3x2xf32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()

  return
}
