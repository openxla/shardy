// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @parallel_reshape_non_divisible(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>})
  -> (tensor<6x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"b"}, {}]>]>} : (tensor<24xi32>) -> tensor<6x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<6x4xi32>
  return %1 : tensor<6x4xi32>
}

func.func @sequential_reshape_non_divisible(%arg0: tensor<24xi32>) -> tensor<6x4xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<24xi32>) -> tensor<6x4xi32>
  return %0 : tensor<6x4xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[
    1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12,
    13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24
  ]> : tensor<24xi32>

  %seq = func.call @sequential_reshape_non_divisible(%input) : (tensor<24xi32>) -> tensor<6x4xi32>

  // Slice input into 2 shards of size 12.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 12>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<12xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 12>, limit_indices = array<i64: 24>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<12xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s0, %s1) {
    programs = [[@parallel_reshape_non_divisible, @parallel_reshape_non_divisible, @parallel_reshape_non_divisible, @parallel_reshape_non_divisible]]
  } : (tensor<12xi32>, tensor<12xi32>, tensor<12xi32>, tensor<12xi32>) -> (tensor<6x4xi32>, tensor<6x4xi32>, tensor<6x4xi32>, tensor<6x4xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<6x4xi32>, tensor<6x4xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<6x4xi32>, tensor<6x4xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<6x4xi32>, tensor<6x4xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<6x4xi32>, tensor<6x4xi32>) -> ()

  return
}
