// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_pad_replicated_negative_low_padding(
  %arg0: tensor<8x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
  -> (tensor<12x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [4, -1], high = [0, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x3xi32>, tensor<i32>) -> tensor<12x2xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<12x2xi32>
  return %1 : tensor<12x2xi32>
}

func.func @sequential_pad_replicated_negative_low_padding(%arg0: tensor<8x3xi32>, %arg1: tensor<i32>) -> tensor<12x2xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [4, -1], high = [0, 0], interior = [0, 0] : (tensor<8x3xi32>, tensor<i32>) -> tensor<12x2xi32>
  return %0 : tensor<12x2xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %input_seq = stablehlo.iota dim = 0 : tensor<24xi32>
  %input = stablehlo.reshape %input_seq : (tensor<24xi32>) -> tensor<8x3xi32>

  %seq = func.call @sequential_pad_replicated_negative_low_padding(%input, %c) : (tensor<8x3xi32>, tensor<i32>) -> tensor<12x2xi32>

  // The input is sharded into 2 sub-tensors of size 4x3.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 3>, strides = array<i64: 1, 1>} : (tensor<8x3xi32>) -> tensor<4x3xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 3>, strides = array<i64: 1, 1>} : (tensor<8x3xi32>) -> tensor<4x3xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_pad_replicated_negative_low_padding, @parallel_pad_replicated_negative_low_padding]]
  } : (tensor<4x3xi32>, tensor<4x3xi32>) -> (tensor<12x2xi32>, tensor<12x2xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<12x2xi32>, tensor<12x2xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<12x2xi32>, tensor<12x2xi32>) -> ()

  return
}
