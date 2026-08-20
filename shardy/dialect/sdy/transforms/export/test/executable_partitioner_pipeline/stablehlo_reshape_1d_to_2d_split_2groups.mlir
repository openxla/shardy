// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true

//--- part1.mlir
sdy.mesh @mesh_ab_16 = <["a"=4, "b"=4]>

func.func @parallel_reshape_1d_to_2d_split_2groups(%arg0: tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_ab_16, [{"a"}, {"b"}]>}) -> (tensor<2x3x2x7xi32> {sdy.sharding = #sdy.sharding<@mesh_ab_16, [{}, {}, {}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6, 0:14] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_ab_16, [{"a"}, {"b"}]>]>} : (tensor<8x16xi32>) -> tensor<6x14xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_ab_16, [{"a":(1)2}, {"a":(2)2}, {"b":(1)2}, {"b":(2)2}]>]>} : (tensor<6x14xi32>) -> tensor<2x3x2x7xi32>
  %2 = sdy.reshard %1 <@mesh_ab_16, [{}, {}, {}, {}]> : tensor<2x3x2x7xi32>
  return %2 : tensor<2x3x2x7xi32>
}

func.func @sequential_reshape_1d_to_2d_split_2groups(%arg0: tensor<8x16xi32>) -> tensor<2x3x2x7xi32> {
  %0 = stablehlo.slice %arg0 [0:6, 0:14] : (tensor<8x16xi32>) -> tensor<6x14xi32>
  %1 = stablehlo.reshape %0 : (tensor<6x14xi32>) -> tensor<2x3x2x7xi32>
  return %1 : tensor<2x3x2x7xi32>
}

//--- part2.mlir
func.func @main() {
  %input2d_seq = stablehlo.iota dim = 0 : tensor<8x16xi32>
  %c1_2d = stablehlo.constant dense<1> : tensor<8x16xi32>
  %input2d = stablehlo.add %input2d_seq, %c1_2d : tensor<8x16xi32>
  %seq2 = func.call @sequential_reshape_1d_to_2d_split_2groups(%input2d) : (tensor<8x16xi32>) -> tensor<2x3x2x7xi32>

  %d0_0 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d0_1 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d0_2 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 0, 8>, limit_indices = array<i64: 2, 12>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d0_3 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 0, 12>, limit_indices = array<i64: 2, 16>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>

  %d1_0 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d1_1 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 2, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d1_2 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 2, 8>, limit_indices = array<i64: 4, 12>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d1_3 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 2, 12>, limit_indices = array<i64: 4, 16>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>

  %d2_0 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 6, 4>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d2_1 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 4, 4>, limit_indices = array<i64: 6, 8>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d2_2 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 4, 8>, limit_indices = array<i64: 6, 12>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d2_3 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 4, 12>, limit_indices = array<i64: 6, 16>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>

  %d3_0 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d3_1 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 6, 4>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d3_2 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 6, 8>, limit_indices = array<i64: 8, 12>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>
  %d3_3 = "stablehlo.slice"(%input2d) {start_indices = array<i64: 6, 12>, limit_indices = array<i64: 8, 16>, strides = array<i64: 1, 1>} : (tensor<8x16xi32>) -> tensor<2x4xi32>

  %res2:16 = "interpreter.run_parallel"(%d0_0, %d0_1, %d0_2, %d0_3, %d1_0, %d1_1, %d1_2, %d1_3, %d2_0, %d2_1, %d2_2, %d2_3, %d3_0, %d3_1, %d3_2, %d3_3) {
    programs = [[@parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups, @parallel_reshape_1d_to_2d_split_2groups]]
  } : (tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>) ->
      (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>)

  "check.expect_eq"(%res2#0, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#1, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#2, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#3, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#4, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#5, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#6, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#7, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#8, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#9, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#10, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#11, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#12, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#13, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#14, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  "check.expect_eq"(%res2#15, %seq2) : (tensor<2x3x2x7xi32>, tensor<2x3x2x7xi32>) -> ()
  return
}
