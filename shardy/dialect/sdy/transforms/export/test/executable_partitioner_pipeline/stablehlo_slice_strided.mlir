// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true

//--- part1.mlir

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @parallel_slice_strided(
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
  -> (tensor<3x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [1:7:2, 0:4]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {}]>]>} : (tensor<8x4xi32>) -> tensor<3x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<3x4xi32>
  return %1 : tensor<3x4xi32>
}

func.func @sequential_slice_strided(%arg0: tensor<8x4xi32>) -> tensor<3x4xi32> {
  %0 = stablehlo.slice %arg0 [1:7:2, 0:4] : (tensor<8x4xi32>) -> tensor<3x4xi32>
  return %0 : tensor<3x4xi32>
}

func.func @parallel_slice_strided_cross_shard_2d(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<3x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [2:8:2, 1:8:3]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>} : (tensor<8x8xi32>) -> tensor<3x3xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<3x3xi32>
  return %1 : tensor<3x3xi32>
}

func.func @sequential_slice_strided_cross_shard_2d(%arg0: tensor<8x8xi32>) -> tensor<3x3xi32> {
  %0 = stablehlo.slice %arg0 [2:8:2, 1:8:3] : (tensor<8x8xi32>) -> tensor<3x3xi32>
  return %0 : tensor<3x3xi32>
}

func.func @parallel_slice_comm_free_and_strided_2d(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<1x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:1:1, 0:8:2]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>} : (tensor<8x8xi32>) -> tensor<1x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<1x4xi32>
  return %1 : tensor<1x4xi32>
}

func.func @sequential_slice_comm_free_and_strided_2d(%arg0: tensor<8x8xi32>) -> tensor<1x4xi32> {
  %0 = stablehlo.slice %arg0 [0:1:1, 0:8:2] : (tensor<8x8xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

//--- part2.mlir

func.func @main() {
  %input1_seq = stablehlo.iota dim = 0 : tensor<32xi32>
  %input1 = stablehlo.reshape %input1_seq : (tensor<32xi32>) -> tensor<8x4xi32>
  %seq1 = func.call @sequential_slice_strided(%input1) : (tensor<8x4xi32>) -> tensor<3x4xi32>

  // @mesh = <["a"=2, "b"=2]>: dim 0 sharded on "a"=2, dim 1 replicated on "b"=2.
  %s1_0 = "stablehlo.slice"(%input1) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<4x4xi32>
  %s1_1 = "stablehlo.slice"(%input1) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<4x4xi32>

  %res1:4 = "interpreter.run_parallel"(%s1_0, %s1_0, %s1_1, %s1_1) {
    programs = [[@parallel_slice_strided, @parallel_slice_strided, @parallel_slice_strided, @parallel_slice_strided]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>)

  "check.expect_eq"(%res1#0, %seq1) : (tensor<3x4xi32>, tensor<3x4xi32>) -> ()
  "check.expect_eq"(%res1#1, %seq1) : (tensor<3x4xi32>, tensor<3x4xi32>) -> ()
  "check.expect_eq"(%res1#2, %seq1) : (tensor<3x4xi32>, tensor<3x4xi32>) -> ()
  "check.expect_eq"(%res1#3, %seq1) : (tensor<3x4xi32>, tensor<3x4xi32>) -> ()

  %input2_seq = stablehlo.iota dim = 0 : tensor<64xi32>
  %input2 = stablehlo.reshape %input2_seq : (tensor<64xi32>) -> tensor<8x8xi32>

  // 4 shards of size 4x4 across @mesh = <["a"=2, "b"=2]>.
  %s2_00 = "stablehlo.slice"(%input2) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %s2_01 = "stablehlo.slice"(%input2) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %s2_10 = "stablehlo.slice"(%input2) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>
  %s2_11 = "stablehlo.slice"(%input2) {start_indices = array<i64: 4, 4>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>} : (tensor<8x8xi32>) -> tensor<4x4xi32>

  // Cross-shard 2D strided slice [2:8:2, 1:8:3] exercising collective_permute
  // HALO exchange on both sharded axes.
  %seq2 = func.call @sequential_slice_strided_cross_shard_2d(%input2) : (tensor<8x8xi32>) -> tensor<3x3xi32>
  %res2:4 = "interpreter.run_parallel"(%s2_00, %s2_01, %s2_10, %s2_11) {
    programs = [[@parallel_slice_strided_cross_shard_2d, @parallel_slice_strided_cross_shard_2d, @parallel_slice_strided_cross_shard_2d, @parallel_slice_strided_cross_shard_2d]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>)

  "check.expect_eq"(%res2#0, %seq2) : (tensor<3x3xi32>, tensor<3x3xi32>) -> ()
  "check.expect_eq"(%res2#1, %seq2) : (tensor<3x3xi32>, tensor<3x3xi32>) -> ()
  "check.expect_eq"(%res2#2, %seq2) : (tensor<3x3xi32>, tensor<3x3xi32>) -> ()
  "check.expect_eq"(%res2#3, %seq2) : (tensor<3x3xi32>, tensor<3x3xi32>) -> ()

  // Communication-free [0:1:1] sharded dim coexisting with strided [0:8:2]
  // sharded dim.
  %seq3 = func.call @sequential_slice_comm_free_and_strided_2d(%input2) : (tensor<8x8xi32>) -> tensor<1x4xi32>
  %res3:4 = "interpreter.run_parallel"(%s2_00, %s2_01, %s2_10, %s2_11) {
    programs = [[@parallel_slice_comm_free_and_strided_2d, @parallel_slice_comm_free_and_strided_2d, @parallel_slice_comm_free_and_strided_2d, @parallel_slice_comm_free_and_strided_2d]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)

  "check.expect_eq"(%res3#0, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res3#1, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res3#2, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res3#3, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()

  return
}
