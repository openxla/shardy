// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_a2_b2 = <["a"=2, "b"=2]>

// 1. Broadcast existing sharded dimensions into a higher-rank tensor.
func.func @parallel_broadcast_in_dim(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"a"}, {"b"}]>})
  -> (tensor<3x4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}, {}]>}) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{}, {"a"}, {"b"}]>]>}
    : (tensor<4x8xi32>) -> tensor<3x4x8xi32>
  %1 = sdy.reshard %0 <@mesh_a2_b2, [{}, {}, {}]> : tensor<3x4x8xi32>
  return %1 : tensor<3x4x8xi32>
}

func.func @sequential_broadcast_in_dim(%arg0: tensor<4x8xi32>) -> tensor<3x4x8xi32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<4x8xi32>) -> tensor<3x4x8xi32>
  return %0 : tensor<3x4x8xi32>
}

// 2. Broadcast into a newly-introduced sharded dimension.
func.func @parallel_broadcast_new_sharded_dim(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{"b"}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a2_b2, [{}, {}]>}) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a2_b2, [{"a"}, {"b"}]>]>}
    : (tensor<8xi32>) -> tensor<4x8xi32>
  %1 = sdy.reshard %0 <@mesh_a2_b2, [{}, {}]> : tensor<4x8xi32>
  return %1 : tensor<4x8xi32>
}

func.func @sequential_broadcast_new_sharded_dim(%arg0: tensor<8xi32>) -> tensor<4x8xi32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<8xi32>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

//--- part2.mlir

func.func @main() {
  // Test 1: Broadcast existing sharded dimensions
  %input_seq = stablehlo.iota dim = 0 : tensor<32xi32>
  %input = stablehlo.reshape %input_seq : (tensor<32xi32>) -> tensor<4x8xi32>

  %seq1 = func.call @sequential_broadcast_in_dim(%input) : (tensor<4x8xi32>) -> tensor<3x4x8xi32>

  // The input is sharded into 4 sub-tensors of size 2x4.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 2, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 4>, limit_indices = array<i64: 4, 8>, strides = array<i64: 1, 1>} : (tensor<4x8xi32>) -> tensor<2x4xi32>

  %res1:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@parallel_broadcast_in_dim, @parallel_broadcast_in_dim, @parallel_broadcast_in_dim, @parallel_broadcast_in_dim]]
  } : (tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>) -> (tensor<3x4x8xi32>, tensor<3x4x8xi32>, tensor<3x4x8xi32>, tensor<3x4x8xi32>)

  "check.expect_eq"(%res1#0, %seq1) : (tensor<3x4x8xi32>, tensor<3x4x8xi32>) -> ()
  "check.expect_eq"(%res1#1, %seq1) : (tensor<3x4x8xi32>, tensor<3x4x8xi32>) -> ()
  "check.expect_eq"(%res1#2, %seq1) : (tensor<3x4x8xi32>, tensor<3x4x8xi32>) -> ()
  "check.expect_eq"(%res1#3, %seq1) : (tensor<3x4x8xi32>, tensor<3x4x8xi32>) -> ()

  // Test 2: Broadcast into newly-introduced sharded dimension
  %input2 = stablehlo.iota dim = 0 : tensor<8xi32>
  %seq2 = func.call @sequential_broadcast_new_sharded_dim(%input2) : (tensor<8xi32>) -> tensor<4x8xi32>

  // The input is sharded by b=2, replicated along a=2.
  // Devices (0,0) and (1,0) hold slice [0:4]; devices (0,1) and (1,1) hold slice [4:8].
  %s2_0 = "stablehlo.slice"(%input2) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>
  %s2_1 = "stablehlo.slice"(%input2) {start_indices = array<i64: 4>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<4xi32>

  %res2:4 = "interpreter.run_parallel"(%s2_0, %s2_1, %s2_0, %s2_1) {
    programs = [[@parallel_broadcast_new_sharded_dim, @parallel_broadcast_new_sharded_dim, @parallel_broadcast_new_sharded_dim, @parallel_broadcast_new_sharded_dim]]
  } : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)

  "check.expect_eq"(%res2#0, %seq2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res2#1, %seq2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res2#2, %seq2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res2#3, %seq2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()

  return
}
