// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true

//--- part1.mlir
sdy.mesh @mesh_custom = <["b"=2, "c"=2], device_ids=[3, 2, 1, 0]>

func.func @parallel_reshape_1d_to_2d_split_custom(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{"b", "c"}]>}) -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_custom, [{"b", "c"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_custom, [{"b"}, {"c"}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
  %2 = sdy.reshard %1 <@mesh_custom, [{}, {}]> : tensor<2x3xi32>
  return %2 : tensor<2x3xi32>
}

func.func @sequential_reshape_1d_to_2d_split_custom(%arg0: tensor<8xi32>) -> tensor<2x3xi32> {
  %0 = stablehlo.slice %arg0 [0:6] : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 : (tensor<6xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

//--- part2.mlir
func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<8xi32>
  %c1 = stablehlo.constant dense<1> : tensor<8xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<8xi32>
  %seq = func.call @sequential_reshape_1d_to_2d_split_custom(%input) : (tensor<8xi32>) -> tensor<2x3xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 6>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %res:4 = "interpreter.run_parallel"(%s3, %s2, %s1, %s0) {
    programs = [[@parallel_reshape_1d_to_2d_split_custom, @parallel_reshape_1d_to_2d_split_custom, @parallel_reshape_1d_to_2d_split_custom, @parallel_reshape_1d_to_2d_split_custom]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) ->
      (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>)
  "check.expect_eq"(%res#0, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  return
}
