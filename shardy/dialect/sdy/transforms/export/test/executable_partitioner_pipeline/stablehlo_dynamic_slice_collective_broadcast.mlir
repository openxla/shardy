// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh = <["x"=4]>
sdy.mesh @mesh_2x2 = <["a"=2, "b"=2]>

func.func @parallel_dynamic_slice_collective_broadcast(
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
  %arg1: tensor<i32>)
  -> (tensor<1x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %c0, sizes = [1, 4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

func.func @parallel_dynamic_slice_transposed_mesh(
  %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2x2, [{"b", "a"}, {}]>},
  %arg1: tensor<i32>)
  -> (tensor<1x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2x2, [{}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %c0, sizes = [1, 4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2x2, [{}, {}]>]>} : (tensor<8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

func.func @sequential_dynamic_slice_collective_broadcast(
  %arg0: tensor<8x4xi32>,
  %arg1: tensor<i32>) -> tensor<1x4xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %c0, sizes = [1, 4] : (tensor<8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

//--- part2.mlir

func.func @main() {
  %input_seq = stablehlo.iota dim = 0 : tensor<32xi32>
  %input = stablehlo.reshape %input_seq : (tensor<32xi32>) -> tensor<8x4xi32>

  // The input is sharded into 4 sub-tensors of size 2x4.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<2x4xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<2x4xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 4, 0>, limit_indices = array<i64: 6, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<2x4xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>} : (tensor<8x4xi32>) -> tensor<2x4xi32>

  // Case 1: In-bounds start index 5 (lands on shard 2, local index 1).
  %idx5 = stablehlo.constant dense<5> : tensor<i32>
  %seq5 = func.call @sequential_dynamic_slice_collective_broadcast(%input, %idx5) : (tensor<8x4xi32>, tensor<i32>) -> tensor<1x4xi32>
  %res5:4 = "interpreter.run_parallel"(%s0, %idx5, %s1, %idx5, %s2, %idx5, %s3, %idx5) {
    programs = [[@parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast]]
  } : (tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>) -> (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)
  "check.expect_eq"(%res5#0, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res5#1, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res5#2, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res5#3, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()

  // Case 2: In-bounds start index 3 (lands on shard 1, local index 1).
  %idx3 = stablehlo.constant dense<3> : tensor<i32>
  %seq3 = func.call @sequential_dynamic_slice_collective_broadcast(%input, %idx3) : (tensor<8x4xi32>, tensor<i32>) -> tensor<1x4xi32>
  %res3:4 = "interpreter.run_parallel"(%s0, %idx3, %s1, %idx3, %s2, %idx3, %s3, %idx3) {
    programs = [[@parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast]]
  } : (tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>) -> (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)
  "check.expect_eq"(%res3#0, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res3#1, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res3#2, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res3#3, %seq3) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()

  // Case 3: Out-of-bounds negative start index -3 (clamped to 0 on shard 0).
  %idx_neg = stablehlo.constant dense<-3> : tensor<i32>
  %seq_neg = func.call @sequential_dynamic_slice_collective_broadcast(%input, %idx_neg) : (tensor<8x4xi32>, tensor<i32>) -> tensor<1x4xi32>
  %res_neg:4 = "interpreter.run_parallel"(%s0, %idx_neg, %s1, %idx_neg, %s2, %idx_neg, %s3, %idx_neg) {
    programs = [[@parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast]]
  } : (tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>) -> (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)
  "check.expect_eq"(%res_neg#0, %seq_neg) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_neg#1, %seq_neg) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_neg#2, %seq_neg) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_neg#3, %seq_neg) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()

  // Case 4: Out-of-bounds high start index 12 (clamped to 7 on shard 3, local index 1).
  %idx_high = stablehlo.constant dense<12> : tensor<i32>
  %seq_high = func.call @sequential_dynamic_slice_collective_broadcast(%input, %idx_high) : (tensor<8x4xi32>, tensor<i32>) -> tensor<1x4xi32>
  %res_high:4 = "interpreter.run_parallel"(%s0, %idx_high, %s1, %idx_high, %s2, %idx_high, %s3, %idx_high) {
    programs = [[@parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast, @parallel_dynamic_slice_collective_broadcast]]
  } : (tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>) -> (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)
  "check.expect_eq"(%res_high#0, %seq_high) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_high#1, %seq_high) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_high#2, %seq_high) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_high#3, %seq_high) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()

  // Case 5: Transposed multi-axis sharding [{"b", "a"}, {}] on @mesh_2x2 = <["a"=2, "b"=2]>,
  // where dev 0 holds shard 0 (%s0), dev 1 holds shard 2 (%s2), dev 2 holds shard 1 (%s1), dev 3 holds shard 3 (%s3).
  %res_transposed:4 = "interpreter.run_parallel"(%s0, %idx5, %s2, %idx5, %s1, %idx5, %s3, %idx5) {
    programs = [[@parallel_dynamic_slice_transposed_mesh, @parallel_dynamic_slice_transposed_mesh, @parallel_dynamic_slice_transposed_mesh, @parallel_dynamic_slice_transposed_mesh]]
  } : (tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>, tensor<2x4xi32>, tensor<i32>) -> (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)
  "check.expect_eq"(%res_transposed#0, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_transposed#1, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_transposed#2, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res_transposed#3, %seq5) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()

  return
}
