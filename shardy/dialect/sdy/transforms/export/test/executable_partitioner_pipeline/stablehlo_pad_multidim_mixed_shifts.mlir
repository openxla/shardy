// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_pad_multidim_mixed_shifts(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [-2, 1], high = [6, 1], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x2xi32>, tensor<i32>) -> tensor<8x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<8x4xi32>
  return %1 : tensor<8x4xi32>
}

func.func @sequential_pad_multidim_mixed_shifts(%arg0: tensor<4x2xi32>, %arg1: tensor<i32>) -> tensor<8x4xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [-2, 1], high = [6, 1], interior = [0, 0] : (tensor<4x2xi32>, tensor<i32>) -> tensor<8x4xi32>
  return %0 : tensor<8x4xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %input_seq = stablehlo.iota dim = 0 : tensor<8xi32>
  %input = stablehlo.reshape %input_seq : (tensor<8xi32>) -> tensor<4x2xi32>

  %seq = func.call @sequential_pad_multidim_mixed_shifts(%input, %c) : (tensor<4x2xi32>, tensor<i32>) -> tensor<8x4xi32>

  // The input is sharded into 2 sub-tensors of size 2x2.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<2x2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_pad_multidim_mixed_shifts, @parallel_pad_multidim_mixed_shifts]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<8x4xi32>, tensor<8x4xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<8x4xi32>, tensor<8x4xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<8x4xi32>, tensor<8x4xi32>) -> ()

  return
}
