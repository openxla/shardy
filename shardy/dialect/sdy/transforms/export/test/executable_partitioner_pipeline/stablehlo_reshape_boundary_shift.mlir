// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["a"=4]>

func.func @parallel_reshape_boundary_shift(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {"a"}]>]>} : (tensor<24xi32>) -> tensor<4x6xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {"a"}]> : tensor<4x6xi32>
  %2 = sdy.reshard %1 <@mesh, [{}, {}]> : tensor<4x6xi32>
  return %2 : tensor<4x6xi32>
}

func.func @sequential_reshape_boundary_shift(%arg0: tensor<24xi32>) -> tensor<4x6xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<24xi32>) -> tensor<4x6xi32>
  return %0 : tensor<4x6xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[
    1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12,
    13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24
  ]> : tensor<24xi32>

  %seq = func.call @sequential_reshape_boundary_shift(%input) : (tensor<24xi32>) -> tensor<4x6xi32>

  // Slice input into 4 shards of local size 6.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 6>, limit_indices = array<i64: 12>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 12>, limit_indices = array<i64: 18>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 18>, limit_indices = array<i64: 24>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_reshape_boundary_shift, @parallel_reshape_boundary_shift, @parallel_reshape_boundary_shift, @parallel_reshape_boundary_shift]]
  } : (tensor<6xi32>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<4x6xi32>, tensor<4x6xi32>, tensor<4x6xi32>, tensor<4x6xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<4x6xi32>, tensor<4x6xi32>) -> ()

  return
}
