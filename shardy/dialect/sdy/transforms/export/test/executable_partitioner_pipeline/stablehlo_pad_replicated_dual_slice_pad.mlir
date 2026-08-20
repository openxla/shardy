// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @parallel_pad_replicated_dual_slice_pad(
  %arg0: tensor<8x8x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>})
  -> (tensor<11x8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [3, 0, -1], high = [0, 0, 2], interior = [0, 0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}, {}]>]>} : (tensor<8x8x3xi32>, tensor<i32>) -> tensor<11x8x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {}]> : tensor<11x8x4xi32>
  return %1 : tensor<11x8x4xi32>
}

func.func @sequential_pad_replicated_dual_slice_pad(%arg0: tensor<8x8x3xi32>, %arg1: tensor<i32>) -> tensor<11x8x4xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [3, 0, -1], high = [0, 0, 2], interior = [0, 0, 0] : (tensor<8x8x3xi32>, tensor<i32>) -> tensor<11x8x4xi32>
  return %0 : tensor<11x8x4xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %input_seq = stablehlo.iota dim = 0 : tensor<192xi32>
  %input = stablehlo.reshape %input_seq : (tensor<192xi32>) -> tensor<8x8x3xi32>

  %seq = func.call @sequential_pad_replicated_dual_slice_pad(%input, %c) : (tensor<8x8x3xi32>, tensor<i32>) -> tensor<11x8x4xi32>

  // The input is sharded into 4 sub-tensors of size 4x4x3.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: 4, 4, 3>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x3xi32>) -> tensor<4x4x3xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 4, 0>, limit_indices = array<i64: 4, 8, 3>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x3xi32>) -> tensor<4x4x3xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0, 0>, limit_indices = array<i64: 8, 4, 3>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x3xi32>) -> tensor<4x4x3xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 4, 0>, limit_indices = array<i64: 8, 8, 3>, strides = array<i64: 1, 1, 1>} : (tensor<8x8x3xi32>) -> tensor<4x4x3xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_pad_replicated_dual_slice_pad, @parallel_pad_replicated_dual_slice_pad, @parallel_pad_replicated_dual_slice_pad, @parallel_pad_replicated_dual_slice_pad]]
  } : (tensor<4x4x3xi32>, tensor<4x4x3xi32>, tensor<4x4x3xi32>, tensor<4x4x3xi32>) -> (tensor<11x8x4xi32>, tensor<11x8x4xi32>, tensor<11x8x4xi32>, tensor<11x8x4xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<11x8x4xi32>, tensor<11x8x4xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<11x8x4xi32>, tensor<11x8x4xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<11x8x4xi32>, tensor<11x8x4xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<11x8x4xi32>, tensor<11x8x4xi32>) -> ()

  return
}
