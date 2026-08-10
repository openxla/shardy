// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"

//--- part1.mlir
sdy.mesh @mesh_bc_6 = <["b"=2, "c"=3]>

func.func @parallel_reshape_1d_to_2d_split_gap_2(%arg0: tensor<18xi32> {sdy.sharding = #sdy.sharding<@mesh_bc_6, [{"b", "c"}]>}) -> (tensor<2x7xi32> {sdy.sharding = #sdy.sharding<@mesh_bc_6, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:14] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_bc_6, [{"b", "c"}]>]>} : (tensor<18xi32>) -> tensor<14xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_bc_6, [{"b"}, {"c"}]>]>} : (tensor<14xi32>) -> tensor<2x7xi32>
  %2 = sdy.reshard %1 <@mesh_bc_6, [{}, {}]> : tensor<2x7xi32>
  return %2 : tensor<2x7xi32>
}

func.func @sequential_reshape_1d_to_2d_split_gap_2(%arg0: tensor<18xi32>) -> tensor<2x7xi32> {
  %0 = stablehlo.slice %arg0 [0:14] : (tensor<18xi32>) -> tensor<14xi32>
  %1 = stablehlo.reshape %0 : (tensor<14xi32>) -> tensor<2x7xi32>
  return %1 : tensor<2x7xi32>
}

//--- part2.mlir
func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<18xi32>
  %c1 = stablehlo.constant dense<1> : tensor<18xi32>
  %input = stablehlo.add %input_seq, %c1 : tensor<18xi32>
  %seq = func.call @sequential_reshape_1d_to_2d_split_gap_2(%input) : (tensor<18xi32>) -> tensor<2x7xi32>
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 3>, strides = array<i64: 1>} : (tensor<18xi32>) -> tensor<3xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 3>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<18xi32>) -> tensor<3xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 6>, limit_indices = array<i64: 9>, strides = array<i64: 1>} : (tensor<18xi32>) -> tensor<3xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 9>, limit_indices = array<i64: 12>, strides = array<i64: 1>} : (tensor<18xi32>) -> tensor<3xi32>
  %s4 = "stablehlo.slice"(%input) {start_indices = array<i64: 12>, limit_indices = array<i64: 15>, strides = array<i64: 1>} : (tensor<18xi32>) -> tensor<3xi32>
  %s5 = "stablehlo.slice"(%input) {start_indices = array<i64: 15>, limit_indices = array<i64: 18>, strides = array<i64: 1>} : (tensor<18xi32>) -> tensor<3xi32>
  %res:6 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3, %s4, %s5) {
    programs = [[@parallel_reshape_1d_to_2d_split_gap_2,
                 @parallel_reshape_1d_to_2d_split_gap_2,
                 @parallel_reshape_1d_to_2d_split_gap_2,
                 @parallel_reshape_1d_to_2d_split_gap_2,
                 @parallel_reshape_1d_to_2d_split_gap_2,
                 @parallel_reshape_1d_to_2d_split_gap_2]]
  } : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) ->
      (tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>)
  "check.expect_eq"(%res#0, %seq) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#4, %seq) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#5, %seq) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  return
}
