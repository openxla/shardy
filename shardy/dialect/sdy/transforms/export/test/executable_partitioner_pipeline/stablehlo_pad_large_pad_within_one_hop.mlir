// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

func.func @parallel_pad_large_pad(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {}]>})
  -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [12, 0], high = [0, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<16x8xi32>
  %1 = sdy.reshard %0 <@mesh_a2_b2, [{}, {}]> : tensor<16x8xi32>
  return %1 : tensor<16x8xi32>
}

func.func @sequential_pad_large_pad(%arg0: tensor<4x8xi32>, %arg1: tensor<i32>) -> tensor<16x8xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [12, 0], high = [0, 0], interior = [0, 0] : (tensor<4x8xi32>, tensor<i32>) -> tensor<16x8xi32>
  return %0 : tensor<16x8xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %input_seq = stablehlo.iota dim = 0 : tensor<32xi32>
  %input = stablehlo.reshape %input_seq : (tensor<32xi32>) -> tensor<4x8xi32>

  %seq = func.call @sequential_pad_large_pad(%input, %c) : (tensor<4x8xi32>, tensor<i32>) -> tensor<16x8xi32>

  // The input is sharded into 2 sub-tensors of size 2x8.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x8xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x8xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s0, %s1, %s1) {
    programs = [[@parallel_pad_large_pad, @parallel_pad_large_pad, @parallel_pad_large_pad, @parallel_pad_large_pad]]
  } : (tensor<2x8xi32>, tensor<2x8xi32>, tensor<2x8xi32>, tensor<2x8xi32>) -> (tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>, tensor<16x8xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<16x8xi32>, tensor<16x8xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<16x8xi32>, tensor<16x8xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<16x8xi32>, tensor<16x8xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<16x8xi32>, tensor<16x8xi32>) -> ()

  return
}
