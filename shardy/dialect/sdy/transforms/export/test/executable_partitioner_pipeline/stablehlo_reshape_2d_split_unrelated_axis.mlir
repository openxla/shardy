// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true

//--- part1.mlir
sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

func.func @parallel_reshape_2d_split_unrelated_axis(%arg0: tensor<6x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>}) -> (tensor<1x5x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:5, 0:4] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>]>} : (tensor<6x4xi32>) -> tensor<5x4xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a2_b2, [{"a"}, {}, {"b"}]>]>} : (tensor<5x4xi32>) -> tensor<1x5x4xi32>
  %2 = sdy.reshard %1 <@mesh_a2_b2, [{}, {}, {}]> : tensor<1x5x4xi32>
  return %2 : tensor<1x5x4xi32>
}

func.func @sequential_reshape_2d_split_unrelated_axis(%arg0: tensor<6x4xi32>) -> tensor<1x5x4xi32> {
  %0 = stablehlo.slice %arg0 [0:5, 0:4] : (tensor<6x4xi32>) -> tensor<5x4xi32>
  %1 = stablehlo.reshape %0 : (tensor<5x4xi32>) -> tensor<1x5x4xi32>
  return %1 : tensor<1x5x4xi32>
}

//--- part2.mlir
func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<6x4xi32>
  %c1 = stablehlo.constant dense<1> : tensor<6x4xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<6x4xi32>
  %seq = func.call @sequential_reshape_2d_split_unrelated_axis(%input) : (tensor<6x4xi32>) -> tensor<1x5x4xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 3, 2>, strides = array<i64: 1, 1>} : (tensor<6x4xi32>) -> tensor<3x2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 3, 4>, strides = array<i64: 1, 1>} : (tensor<6x4xi32>) -> tensor<3x2xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 6, 2>, strides = array<i64: 1, 1>} : (tensor<6x4xi32>) -> tensor<3x2xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3, 2>, limit_indices = array<i64: 6, 4>, strides = array<i64: 1, 1>} : (tensor<6x4xi32>) -> tensor<3x2xi32>
  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_reshape_2d_split_unrelated_axis, @parallel_reshape_2d_split_unrelated_axis, @parallel_reshape_2d_split_unrelated_axis, @parallel_reshape_2d_split_unrelated_axis]]
  } : (tensor<3x2xi32>, tensor<3x2xi32>, tensor<3x2xi32>, tensor<3x2xi32>) ->
      (tensor<1x5x4xi32>, tensor<1x5x4xi32>, tensor<1x5x4xi32>, tensor<1x5x4xi32>)
  "check.expect_eq"(%res#0, %seq) : (tensor<1x5x4xi32>, tensor<1x5x4xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<1x5x4xi32>, tensor<1x5x4xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<1x5x4xi32>, tensor<1x5x4xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<1x5x4xi32>, tensor<1x5x4xi32>) -> ()
  return
}
