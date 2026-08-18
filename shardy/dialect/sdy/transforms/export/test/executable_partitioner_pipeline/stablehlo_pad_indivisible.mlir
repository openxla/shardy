// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh_x4 = <["x"=4]>

// The pad input is sliced from 8x2 to 7x2 (indivisible, padded to 8x2).
// The original pad result size (10) is also indivisible by mesh axis size (4).
// PadForDivisibility should adjust high padding of the pad op to 3 (instead of 2)
// to make the result 12x2 (divisible). The result is then trimmed to 10x2
// after the final reshard (all_gather).
func.func @parallel_pad_extend(
  %arg0: tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{"x"}, {}]>})
  -> (tensor<10x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x4, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:7, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{"x"}, {}]>]>} : (tensor<8x2xf32>) -> tensor<7x2xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [1, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x4, [{"x"}, {}]>]>} : (tensor<7x2xf32>, tensor<f32>) -> tensor<10x2xf32>
  return %1 : tensor<10x2xf32>
}

func.func @sequential_pad_extend(%arg0: tensor<8x2xf32>) -> tensor<10x2xf32> {
  %0 = stablehlo.slice %arg0 [0:7, 0:2] : (tensor<8x2xf32>) -> tensor<7x2xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.pad %0, %cst, low = [1, 0], high = [2, 0], interior = [0, 0] : (tensor<7x2xf32>, tensor<f32>) -> tensor<10x2xf32>
  return %1 : tensor<10x2xf32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[
    [1.02, -2.45],
    [4.03, 5.76],
    [-7.34, 8.91],
    [0.12, -9.54],
    [11.23, -12.45],
    [-13.67, 14.89],
    [15.01, -16.32],
    [-17.54, 18.76]
  ]> : tensor<8x2xf32>

  %seq = func.call @sequential_pad_extend(%input) : (tensor<8x2xf32>) -> tensor<10x2xf32>

  // Slice input into 4 shards of size 2x2.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<8x2xf32>) -> tensor<2x2xf32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<8x2xf32>) -> tensor<2x2xf32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 6, 2>, strides = array<i64: 1, 1>} : (tensor<8x2xf32>) -> tensor<2x2xf32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 8, 2>, strides = array<i64: 1, 1>} : (tensor<8x2xf32>) -> tensor<2x2xf32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_pad_extend, @parallel_pad_extend, @parallel_pad_extend, @parallel_pad_extend]]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<10x2xf32>, tensor<10x2xf32>, tensor<10x2xf32>, tensor<10x2xf32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<10x2xf32>, tensor<10x2xf32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<10x2xf32>, tensor<10x2xf32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<10x2xf32>, tensor<10x2xf32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<10x2xf32>, tensor<10x2xf32>) -> ()

  return
}
