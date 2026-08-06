// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["a"=4]>

func.func @parallel_reshape_subaxis(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a":(1)2}, {"a":(2)2}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
  %2 = sdy.reshard %1 <@mesh, [{}, {}]> : tensor<2x3xi32>
  return %2 : tensor<2x3xi32>
}

func.func @sequential_reshape_subaxis(%arg0: tensor<8xi32>) -> tensor<2x3xi32> {
  %0 = stablehlo.slice %arg0 [0:6] : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 : (tensor<6xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>

  %seq = func.call @sequential_reshape_subaxis(%input) : (tensor<8xi32>) -> tensor<2x3xi32>

  // Slice input into 4 shards of local size 2.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 6>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_reshape_subaxis, @parallel_reshape_subaxis, @parallel_reshape_subaxis, @parallel_reshape_subaxis]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()

  return
}
