// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>

func.func @parallel_pad_left_shift(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.pad %arg0, %c, low = [-2], high = [6], interior = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  %1 = sdy.reshard %0 <@mesh, [{}]> : tensor<8xi32>
  return %1 : tensor<8xi32>
}

func.func @sequential_pad_left_shift(%arg0: tensor<4xi32>, %arg1: tensor<i32>) -> tensor<8xi32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [-2], high = [6], interior = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

//--- part2.mlir

func.func @main() {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  %seq = func.call @sequential_pad_left_shift(%input, %c) : (tensor<4xi32>, tensor<i32>) -> tensor<8xi32>

  // Slice input into 2 shards of size 2.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_pad_left_shift, @parallel_pad_left_shift]]
  } : (tensor<2xi32>, tensor<2xi32>) -> (tensor<8xi32>, tensor<8xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<8xi32>, tensor<8xi32>) -> ()

  return
}
