// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["a"=4]>

func.func @parallel_reshape_axis_shift(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> (tensor<3x6xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:18] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>} : (tensor<24xi32>) -> tensor<18xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {"a"}]>]>} : (tensor<18xi32>) -> tensor<3x6xi32>
  %2 = sdy.reshard %1 <@mesh, [{}, {}]> : tensor<3x6xi32>
  return %2 : tensor<3x6xi32>
}

func.func @sequential_reshape_axis_shift(%arg0: tensor<24xi32>) -> tensor<3x6xi32> {
  %0 = stablehlo.slice %arg0 [0:18] : (tensor<24xi32>) -> tensor<18xi32>
  %1 = stablehlo.reshape %0 : (tensor<18xi32>) -> tensor<3x6xi32>
  return %1 : tensor<3x6xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<24xi32>
  %c1 = stablehlo.constant dense<1> : tensor<24xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<24xi32>

  %seq = func.call @sequential_reshape_axis_shift(%input) : (tensor<24xi32>) -> tensor<3x6xi32>

  // Slice input into 4 shards of local size 6.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 6>, limit_indices = array<i64: 12>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 12>, limit_indices = array<i64: 18>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 18>, limit_indices = array<i64: 24>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_reshape_axis_shift, @parallel_reshape_axis_shift, @parallel_reshape_axis_shift, @parallel_reshape_axis_shift]]
  } : (tensor<6xi32>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()

  return
}
