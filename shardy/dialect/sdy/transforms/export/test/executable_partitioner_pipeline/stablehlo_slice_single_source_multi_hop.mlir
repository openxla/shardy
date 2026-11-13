// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

// Slices that read a single input shard which is more than one hop away from
// an output shard.

//--- part1.mlir

sdy.mesh @mesh_custom = <["a"=2, "b"=2], device_ids=[2, 0, 3, 1]>
sdy.mesh @mesh_a8 = <["a"=8]>
sdy.mesh @mesh_3x3 = <["a"=3, "b"=3]>

// The dim is sharded 4 ways by two axes. The 5 sliced elements are on shard 2
// and the output is on shards 0, 1 and 2.
func.func @parallel_slice_source_among_targets(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{"a", "b"}]>})
  -> (tensor<5xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{}]>}) {
  %0 = stablehlo.slice %arg0 [12:17]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{"a", "b"}]>]>} : (tensor<24xi32>) -> tensor<5xi32>
  %1 = sdy.reshard %0 <@mesh_custom, [{}]> : tensor<5xi32>
  return %1 : tensor<5xi32>
}

func.func @sequential_slice_source_among_targets(%arg0: tensor<24xi32>) -> tensor<5xi32> {
  %0 = stablehlo.slice %arg0 [12:17] : (tensor<24xi32>) -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// Dim 0 is sharded 4 ways by a sub-axis: the sliced row is on shard 3 and the
// output is on shard 0. Dim 1 is sharded 2 ways and needs a one-hop exchange.
func.func @parallel_slice_far_end_and_one_hop_dim(
  %arg0: tensor<12x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{"a":(1)4}, {"a":(4)2}]>})
  -> (tensor<1x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [10:11, 1:5]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{"a":(1)4}, {"a":(4)2}]>]>} : (tensor<12x8xi32>) -> tensor<1x4xi32>
  %1 = sdy.reshard %0 <@mesh_a8, [{}, {}]> : tensor<1x4xi32>
  return %1 : tensor<1x4xi32>
}

func.func @sequential_slice_far_end_and_one_hop_dim(%arg0: tensor<12x8xi32>) -> tensor<1x4xi32> {
  %0 = stablehlo.slice %arg0 [10:11, 1:5] : (tensor<12x8xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// Prefix slice produces an indivisible 21-element tensor on 4 shards (padded
// shard size 6); slicing [18:21] reads all 3 elements from partially padded
// shard 3 and sends to targets 0, 1, and 2 (sourceShard >= numTargets).
func.func @parallel_slice_indivisible_all_targets_remote(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{"a", "b"}]>})
  -> (tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{}]>}) {
  %0 = stablehlo.slice %arg0 [0:21]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{"a", "b"}]>]>} : (tensor<24xi32>) -> tensor<21xi32>
  %1 = stablehlo.slice %0 [18:21]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{"a", "b"}]>]>} : (tensor<21xi32>) -> tensor<3xi32>
  %2 = sdy.reshard %1 <@mesh_custom, [{}]> : tensor<3xi32>
  return %2 : tensor<3xi32>
}

func.func @sequential_slice_indivisible_all_targets_remote(%arg0: tensor<24xi32>) -> tensor<3xi32> {
  %0 = stablehlo.slice %arg0 [0:21] : (tensor<24xi32>) -> tensor<21xi32>
  %1 = stablehlo.slice %0 [18:21] : (tensor<21xi32>) -> tensor<3xi32>
  return %1 : tensor<3xi32>
}

// Slicing [24:32] of 32 elements on 4 shards reads shard 3 and splits evenly
// across all 4 output shards (size == numTargets * sOut, no PadOp).
func.func @parallel_slice_all_targets_no_pad(
  %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{"a", "b"}]>})
  -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{}]>}) {
  %0 = stablehlo.slice %arg0 [24:32]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{"a", "b"}]>]>} : (tensor<32xi32>) -> tensor<8xi32>
  %1 = sdy.reshard %0 <@mesh_custom, [{}]> : tensor<8xi32>
  return %1 : tensor<8xi32>
}

func.func @sequential_slice_all_targets_no_pad(%arg0: tensor<32xi32>) -> tensor<8xi32> {
  %0 = stablehlo.slice %arg0 [24:32] : (tensor<32xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// 2D slice where both sharded dimensions take the single-source multi-hop path.
func.func @parallel_slice_2d_both_single_source(
  %arg0: tensor<9x12xi32> {sdy.sharding = #sdy.sharding<@mesh_3x3, [{"a"}, {"b"}]>})
  -> (tensor<2x1xi32> {sdy.sharding = #sdy.sharding<@mesh_3x3, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [6:8, 9:10]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3x3, [{"a"}, {"b"}]>]>} : (tensor<9x12xi32>) -> tensor<2x1xi32>
  %1 = sdy.reshard %0 <@mesh_3x3, [{}, {}]> : tensor<2x1xi32>
  return %1 : tensor<2x1xi32>
}

func.func @sequential_slice_2d_both_single_source(%arg0: tensor<9x12xi32>) -> tensor<2x1xi32> {
  %0 = stablehlo.slice %arg0 [6:8, 9:10] : (tensor<9x12xi32>) -> tensor<2x1xi32>
  return %0 : tensor<2x1xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.iota dim = 0 : tensor<24xi32>

  %seq = func.call @sequential_slice_source_among_targets(%input) : (tensor<24xi32>) -> tensor<5xi32>

  // The input is sharded into 4 sub-tensors of size 6.
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 6>, limit_indices = array<i64: 12>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 12>, limit_indices = array<i64: 18>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 18>, limit_indices = array<i64: 24>, strides = array<i64: 1>} : (tensor<24xi32>) -> tensor<6xi32>

  // Shard i runs on device device_ids[i].
  %res:4 = "interpreter.run_parallel"(%s1, %s3, %s0, %s2) {
    programs = [[@parallel_slice_source_among_targets, @parallel_slice_source_among_targets, @parallel_slice_source_among_targets, @parallel_slice_source_among_targets]]
  } : (tensor<6xi32>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<5xi32>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>)

  "check.expect_eq"(%res#0, %seq) : (tensor<5xi32>, tensor<5xi32>) -> ()
  "check.expect_eq"(%res#1, %seq) : (tensor<5xi32>, tensor<5xi32>) -> ()
  "check.expect_eq"(%res#2, %seq) : (tensor<5xi32>, tensor<5xi32>) -> ()
  "check.expect_eq"(%res#3, %seq) : (tensor<5xi32>, tensor<5xi32>) -> ()

  %input2_seq = stablehlo.iota dim = 0 : tensor<96xi32>
  %input2 = stablehlo.reshape %input2_seq : (tensor<96xi32>) -> tensor<12x8xi32>

  %seq2 = func.call @sequential_slice_far_end_and_one_hop_dim(%input2) : (tensor<12x8xi32>) -> tensor<1x4xi32>

  // The input is sharded into 8 sub-tensors of size 3x4. Device 2 * i + j holds
  // rows [3 * i, 3 * i + 3) and columns [4 * j, 4 * j + 4).
  %t0 = "stablehlo.slice"(%input2) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 3, 4>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>
  %t1 = "stablehlo.slice"(%input2) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 3, 8>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>
  %t2 = "stablehlo.slice"(%input2) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 6, 4>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>
  %t3 = "stablehlo.slice"(%input2) {start_indices = array<i64: 3, 4>, limit_indices = array<i64: 6, 8>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>
  %t4 = "stablehlo.slice"(%input2) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 9, 4>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>
  %t5 = "stablehlo.slice"(%input2) {start_indices = array<i64: 6, 4>, limit_indices = array<i64: 9, 8>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>
  %t6 = "stablehlo.slice"(%input2) {start_indices = array<i64: 9, 0>, limit_indices = array<i64: 12, 4>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>
  %t7 = "stablehlo.slice"(%input2) {start_indices = array<i64: 9, 4>, limit_indices = array<i64: 12, 8>, strides = array<i64: 1, 1>} : (tensor<12x8xi32>) -> tensor<3x4xi32>

  %res2:8 = "interpreter.run_parallel"(%t0, %t1, %t2, %t3, %t4, %t5, %t6, %t7) {
    programs = [[@parallel_slice_far_end_and_one_hop_dim, @parallel_slice_far_end_and_one_hop_dim, @parallel_slice_far_end_and_one_hop_dim, @parallel_slice_far_end_and_one_hop_dim, @parallel_slice_far_end_and_one_hop_dim, @parallel_slice_far_end_and_one_hop_dim, @parallel_slice_far_end_and_one_hop_dim, @parallel_slice_far_end_and_one_hop_dim]]
  } : (tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>) -> (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)

  "check.expect_eq"(%res2#0, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res2#1, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res2#2, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res2#3, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res2#4, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res2#5, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res2#6, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()
  "check.expect_eq"(%res2#7, %seq2) : (tensor<1x4xi32>, tensor<1x4xi32>) -> ()

  %seq3 = func.call @sequential_slice_indivisible_all_targets_remote(%input) : (tensor<24xi32>) -> tensor<3xi32>
  %res3:4 = "interpreter.run_parallel"(%s1, %s3, %s0, %s2) {
    programs = [[@parallel_slice_indivisible_all_targets_remote, @parallel_slice_indivisible_all_targets_remote, @parallel_slice_indivisible_all_targets_remote, @parallel_slice_indivisible_all_targets_remote]]
  } : (tensor<6xi32>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)
  "check.expect_eq"(%res3#0, %seq3) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res3#1, %seq3) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res3#2, %seq3) : (tensor<3xi32>, tensor<3xi32>) -> ()
  "check.expect_eq"(%res3#3, %seq3) : (tensor<3xi32>, tensor<3xi32>) -> ()

  %input4 = stablehlo.iota dim = 0 : tensor<32xi32>
  %seq4 = func.call @sequential_slice_all_targets_no_pad(%input4) : (tensor<32xi32>) -> tensor<8xi32>
  %u0 = "stablehlo.slice"(%input4) {start_indices = array<i64: 0>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<32xi32>) -> tensor<8xi32>
  %u1 = "stablehlo.slice"(%input4) {start_indices = array<i64: 8>, limit_indices = array<i64: 16>, strides = array<i64: 1>} : (tensor<32xi32>) -> tensor<8xi32>
  %u2 = "stablehlo.slice"(%input4) {start_indices = array<i64: 16>, limit_indices = array<i64: 24>, strides = array<i64: 1>} : (tensor<32xi32>) -> tensor<8xi32>
  %u3 = "stablehlo.slice"(%input4) {start_indices = array<i64: 24>, limit_indices = array<i64: 32>, strides = array<i64: 1>} : (tensor<32xi32>) -> tensor<8xi32>
  %res4:4 = "interpreter.run_parallel"(%u1, %u3, %u0, %u2) {
    programs = [[@parallel_slice_all_targets_no_pad, @parallel_slice_all_targets_no_pad, @parallel_slice_all_targets_no_pad, @parallel_slice_all_targets_no_pad]]
  } : (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>)
  "check.expect_eq"(%res4#0, %seq4) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res4#1, %seq4) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res4#2, %seq4) : (tensor<8xi32>, tensor<8xi32>) -> ()
  "check.expect_eq"(%res4#3, %seq4) : (tensor<8xi32>, tensor<8xi32>) -> ()

  %input5_seq = stablehlo.iota dim = 0 : tensor<108xi32>
  %input5 = stablehlo.reshape %input5_seq : (tensor<108xi32>) -> tensor<9x12xi32>
  %seq5 = func.call @sequential_slice_2d_both_single_source(%input5) : (tensor<9x12xi32>) -> tensor<2x1xi32>
  %m0 = "stablehlo.slice"(%input5) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 3, 4>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m1 = "stablehlo.slice"(%input5) {start_indices = array<i64: 0, 4>, limit_indices = array<i64: 3, 8>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m2 = "stablehlo.slice"(%input5) {start_indices = array<i64: 0, 8>, limit_indices = array<i64: 3, 12>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m3 = "stablehlo.slice"(%input5) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 6, 4>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m4 = "stablehlo.slice"(%input5) {start_indices = array<i64: 3, 4>, limit_indices = array<i64: 6, 8>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m5 = "stablehlo.slice"(%input5) {start_indices = array<i64: 3, 8>, limit_indices = array<i64: 6, 12>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m6 = "stablehlo.slice"(%input5) {start_indices = array<i64: 6, 0>, limit_indices = array<i64: 9, 4>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m7 = "stablehlo.slice"(%input5) {start_indices = array<i64: 6, 4>, limit_indices = array<i64: 9, 8>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %m8 = "stablehlo.slice"(%input5) {start_indices = array<i64: 6, 8>, limit_indices = array<i64: 9, 12>, strides = array<i64: 1, 1>} : (tensor<9x12xi32>) -> tensor<3x4xi32>
  %res5:9 = "interpreter.run_parallel"(%m0, %m1, %m2, %m3, %m4, %m5, %m6, %m7, %m8) {
    programs = [[@parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source, @parallel_slice_2d_both_single_source]]
  } : (tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>, tensor<3x4xi32>) -> (tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>)
  "check.expect_eq"(%res5#0, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#1, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#2, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#3, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#4, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#5, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#6, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#7, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()
  "check.expect_eq"(%res5#8, %seq5) : (tensor<2x1xi32>, tensor<2x1xi32>) -> ()

  return
}
