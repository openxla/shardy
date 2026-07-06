// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @parallel_reverse_divisible(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:8]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>}
    : (tensor<4x8xi32>) -> tensor<4x8xi32>
  %1 = stablehlo.reverse %0, dims = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
    : tensor<4x8xi32>
  %2 = sdy.reshard %1 <@mesh, [{}, {}]> : tensor<4x8xi32>
  return %2 : tensor<4x8xi32>
}

func.func @sequential_reverse_divisible(%arg0: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:8] : (tensor<4x8xi32>) -> tensor<4x8xi32>
  %1 = stablehlo.reverse %0, dims = [0] : tensor<4x8xi32>
  return %1 : tensor<4x8xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<32xi32>
  %input = stablehlo.reshape %input_seq : (tensor<32xi32>) -> tensor<4x8xi32>

  %seq = func.call @sequential_reverse_divisible(%input) : (tensor<4x8xi32>) -> tensor<4x8xi32>

  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>

  %pars:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_reverse_divisible, @parallel_reverse_divisible, @parallel_reverse_divisible, @parallel_reverse_divisible]]
  } : (tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>) ->
      (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)

  check.expect_eq %seq, %pars#0 : tensor<4x8xi32>
  check.expect_eq %seq, %pars#1 : tensor<4x8xi32>

  return
}
