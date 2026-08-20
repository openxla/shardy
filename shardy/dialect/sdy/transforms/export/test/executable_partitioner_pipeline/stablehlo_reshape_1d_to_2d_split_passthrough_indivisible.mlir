// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true

//--- part1.mlir
sdy.mesh @mesh_xbc_8 = <["x"=2, "b"=2, "c"=2]>

func.func @parallel_reshape_1d_to_2d_split_passthrough_indivisible(%arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_xbc_8, [{"x"}, {"b", "c"}]>}) -> (tensor<3x2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_xbc_8, [{}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:3, 0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xbc_8, [{"x"}, {"b", "c"}]>]>} : (tensor<4x8xi32>) -> tensor<3x6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xbc_8, [{"x"}, {"b"}, {"c"}]>]>} : (tensor<3x6xi32>) -> tensor<3x2x3xi32>
  %2 = sdy.reshard %1 <@mesh_xbc_8, [{}, {}, {}]> : tensor<3x2x3xi32>
  return %2 : tensor<3x2x3xi32>
}

func.func @sequential_reshape_1d_to_2d_split_passthrough_indivisible(%arg0: tensor<4x8xi32>) -> tensor<3x2x3xi32> {
  %0 = stablehlo.slice %arg0 [0:3, 0:6] : (tensor<4x8xi32>) -> tensor<3x6xi32>
  %1 = stablehlo.reshape %0 : (tensor<3x6xi32>) -> tensor<3x2x3xi32>
  return %1 : tensor<3x2x3xi32>
}

//--- part2.mlir
func.func @main() {
  %input_seq = stablehlo.iota dim = 1 : tensor<4x8xi32>
  %c1 = stablehlo.constant dense<1> : tensor<4x8xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<4x8xi32>
  %seq = func.call @sequential_reshape_1d_to_2d_split_passthrough_indivisible(%input) : (tensor<4x8xi32>) -> tensor<3x2x3xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 2, 6>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 6>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %s4 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %s5 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %s6 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 4>, limit_indices = array<i64: 4, 6>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %s7 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 6>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x2xi32>
  %res:8 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7) {
    programs = [[@parallel_reshape_1d_to_2d_split_passthrough_indivisible,
                 @parallel_reshape_1d_to_2d_split_passthrough_indivisible,
                 @parallel_reshape_1d_to_2d_split_passthrough_indivisible,
                 @parallel_reshape_1d_to_2d_split_passthrough_indivisible,
                 @parallel_reshape_1d_to_2d_split_passthrough_indivisible,
                 @parallel_reshape_1d_to_2d_split_passthrough_indivisible,
                 @parallel_reshape_1d_to_2d_split_passthrough_indivisible,
                 @parallel_reshape_1d_to_2d_split_passthrough_indivisible]]
  } : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>,
       tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) ->
      (tensor<3x2x3xi32>, tensor<3x2x3xi32>, tensor<3x2x3xi32>, tensor<3x2x3xi32>,
       tensor<3x2x3xi32>, tensor<3x2x3xi32>, tensor<3x2x3xi32>, tensor<3x2x3xi32>)
  "check.expect_eq"(%res#0, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  "check.expect_eq"(%res#4, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  "check.expect_eq"(%res#5, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  "check.expect_eq"(%res#6, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  "check.expect_eq"(%res#7, %seq) : (tensor<3x2x3xi32>, tensor<3x2x3xi32>) -> ()
  return
}
