// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>

func.func @parallel_pad_multiple_hops_left_shift(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [-4, 0], high = [0, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}, {}]> : tensor<4x8xi32>
  return %1 : tensor<4x8xi32>
}

func.func @sequential_pad_multiple_hops_left_shift(%arg0: tensor<8x8xi32>, %arg1: tensor<i32>) -> tensor<4x8xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [-4, 0], high = [0, 0], interior = [0, 0] : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %input_seq = stablehlo.iota dim = 0 : tensor<64xi32>
  %input = stablehlo.reshape %input_seq : (tensor<64xi32>) -> tensor<8x8xi32>

  %seq = func.call @sequential_pad_multiple_hops_left_shift(%input, %c) : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>

  // The input is sharded into 8 sub-tensors of size 2x4.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>
  %s4 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 6, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>
  %s5 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 4>, limit_indices = array<i64: 6, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>
  %s6 = "stablehlo.slice"(%input) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>
  %s7 = "stablehlo.slice"(%input) {start_indices = array<i64: 6, 4>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<2x4xi32>

  %res:8 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7) {
    programs = [[@parallel_pad_multiple_hops_left_shift, @parallel_pad_multiple_hops_left_shift, @parallel_pad_multiple_hops_left_shift, @parallel_pad_multiple_hops_left_shift, @parallel_pad_multiple_hops_left_shift, @parallel_pad_multiple_hops_left_shift, @parallel_pad_multiple_hops_left_shift, @parallel_pad_multiple_hops_left_shift]]
  } : (tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>) -> (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#4, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#5, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#6, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#7, %seq) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()

  return
}
