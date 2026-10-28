// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_4 = <["x"=4]>
sdy.mesh @mesh_2x2 = <["x"=2, "y"=2]>

func.func @parallel_concat(
  %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>},
  %arg1: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>})
  -> (tensor<4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {"x"}]>]>}
    : (tensor<4x2xi32>, tensor<4x4xi32>) -> tensor<4x6xi32>
  return %0 : tensor<4x6xi32>
}

func.func @parallel_rotate_right(
  %arg0: tensor<10xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
  -> (tensor<10xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}]>}) {
  %0 = stablehlo.slice %arg0 [4:10]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>}
    : (tensor<10xi32>) -> tensor<6xi32>
  %1 = stablehlo.slice %arg0 [0:4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>}
    : (tensor<10xi32>) -> tensor<4xi32>
  %2 = stablehlo.concatenate %0, %1, dim = 0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>}
    : (tensor<6xi32>, tensor<4xi32>) -> tensor<10xi32>
  return %2 : tensor<10xi32>
}

func.func @parallel_rotate_right_aligned_padding(
  %arg0: tensor<6xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>})
  -> (tensor<6xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>}) {
  %in = stablehlo.add %arg0, %arg0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : tensor<6xi32>
  %0 = stablehlo.slice %in [4:6]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<6xi32>) -> tensor<2xi32>
  %1 = stablehlo.slice %in [0:4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<6xi32>) -> tensor<4xi32>
  %2 = stablehlo.concatenate %0, %1, dim = 0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<2xi32>, tensor<4xi32>) -> tensor<6xi32>
  return %2 : tensor<6xi32>
}

func.func @parallel_rotate_right_unaligned_padding(
  %arg0: tensor<12xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>})
  -> (tensor<10xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>}) {
  %in = stablehlo.slice %arg0 [0:10]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<12xi32>) -> tensor<10xi32>
  %0 = stablehlo.slice %in [4:10]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<10xi32>) -> tensor<6xi32>
  %1 = stablehlo.slice %in [0:4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<10xi32>) -> tensor<4xi32>
  %2 = stablehlo.concatenate %0, %1, dim = 0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<6xi32>, tensor<4xi32>) -> tensor<10xi32>
  return %2 : tensor<10xi32>
}

func.func @parallel_rotate_right_2d(
  %arg0: tensor<4x10xi32> {sdy.sharding = #sdy.sharding<@mesh_2x2, [{"x"}, {"y"}]>})
  -> (tensor<4x10xi32> {sdy.sharding = #sdy.sharding<@mesh_2x2, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 4:10]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2x2, [{"x"}, {"y"}]>]>}
    : (tensor<4x10xi32>) -> tensor<4x6xi32>
  %1 = stablehlo.slice %arg0 [0:4, 0:4]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2x2, [{"x"}, {"y"}]>]>}
    : (tensor<4x10xi32>) -> tensor<4x4xi32>
  %2 = stablehlo.concatenate %0, %1, dim = 1
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2x2, [{"x"}, {"y"}]>]>}
    : (tensor<4x6xi32>, tensor<4x4xi32>) -> tensor<4x10xi32>
  return %2 : tensor<4x10xi32>
}

func.func @parallel_rotate_right_unaligned_partial_shards(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>})
  -> (tensor<5xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>}) {
  %in = stablehlo.slice %arg0 [0:5]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<8xi32>) -> tensor<5xi32>
  %0 = stablehlo.slice %in [3:5]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<5xi32>) -> tensor<2xi32>
  %1 = stablehlo.slice %in [0:3]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<5xi32>) -> tensor<3xi32>
  %2 = stablehlo.concatenate %0, %1, dim = 0
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>}
    : (tensor<2xi32>, tensor<3xi32>) -> tensor<5xi32>
  return %2 : tensor<5xi32>
}

//--- part2.mlir

func.func @main() {
  %c0 = stablehlo.constant dense<[
    [1, 2], [3, 4],
    [5, 6], [7, 8]
  ]> : tensor<4x2xi32>
  %c1 = stablehlo.constant dense<[
    [10, 20, 30, 40], [50, 60, 70, 80],
    [90, 100, 110, 120], [130, 140, 150, 160]
  ]> : tensor<4x4xi32>

  %seq = func.call @sequential_concat(%c0, %c1) : (tensor<4x2xi32>, tensor<4x4xi32>) -> tensor<4x6xi32>

  // Manually prepare shards for 2 devices (sharded along dimension 1).
  // Device 0: Columns [0:1] of %c0 and [0:2] of %c1
  %s0_0 = "stablehlo.slice"(%c0) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 1>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<4x1xi32>
  %s1_0 = "stablehlo.slice"(%c1) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x2xi32>

  // Device 1: Columns [1:2] of %c0 and [2:4] of %c1
  %s0_1 = "stablehlo.slice"(%c0) {start_indices = array<i64: 0, 1>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<4x1xi32>
  %s1_1 = "stablehlo.slice"(%c1) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x2xi32>

  %res:2 = "interpreter.run_parallel"(%s0_0, %s1_0, %s0_1, %s1_1) {
    programs = [[@parallel_concat, @parallel_concat]]
  } : (tensor<4x1xi32>, tensor<4x2xi32>, tensor<4x1xi32>, tensor<4x2xi32>) ->
      (tensor<4x6xi32>, tensor<4x6xi32>)

  check.expect_eq %seq, %res#0 : tensor<4x6xi32>
  check.expect_eq %seq, %res#1 : tensor<4x6xi32>

  // Test 1: Divisible rotate-right on @mesh_2 (tensor<10xi32>).
  %rot_in = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  %seq_rot = func.call @sequential_rotate_right(%rot_in) : (tensor<10xi32>) -> tensor<10xi32>
  %rot_s0 = "stablehlo.slice"(%rot_in) {start_indices = array<i64: 0>, limit_indices = array<i64: 5>, strides = array<i64: 1>} : (tensor<10xi32>) -> tensor<5xi32>
  %rot_s1 = "stablehlo.slice"(%rot_in) {start_indices = array<i64: 5>, limit_indices = array<i64: 10>, strides = array<i64: 1>} : (tensor<10xi32>) -> tensor<5xi32>
  %rot_res:2 = "interpreter.run_parallel"(%rot_s0, %rot_s1) {
    programs = [[@parallel_rotate_right, @parallel_rotate_right]]
  } : (tensor<5xi32>, tensor<5xi32>) -> (tensor<10xi32>, tensor<10xi32>)
  check.expect_eq %seq_rot, %rot_res#0 : tensor<10xi32>
  check.expect_eq %seq_rot, %rot_res#1 : tensor<10xi32>

  // Test 2: Rotate-right with aligned padding on @mesh_4 (tensor<6xi32>, participatingShards = 3).
  %aligned_in = stablehlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %seq_aligned = func.call @sequential_rotate_right_aligned_padding(%aligned_in) : (tensor<6xi32>) -> tensor<6xi32>
  %aligned_res:4 = "interpreter.run_parallel"(%aligned_in, %aligned_in, %aligned_in, %aligned_in) {
    programs = [[@parallel_rotate_right_aligned_padding, @parallel_rotate_right_aligned_padding, @parallel_rotate_right_aligned_padding, @parallel_rotate_right_aligned_padding]]
  } : (tensor<6xi32>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<6xi32>, tensor<6xi32>, tensor<6xi32>, tensor<6xi32>)
  check.expect_eq %seq_aligned, %aligned_res#0 : tensor<6xi32>
  check.expect_eq %seq_aligned, %aligned_res#1 : tensor<6xi32>
  check.expect_eq %seq_aligned, %aligned_res#2 : tensor<6xi32>
  check.expect_eq %seq_aligned, %aligned_res#3 : tensor<6xi32>

  // Test 3: Rotate-right with unaligned padding on @mesh_4 (tensor<10xi32>, rightPadding = 2).
  %unaligned_in = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200]> : tensor<12xi32>
  %seq_unaligned = func.call @sequential_rotate_right_unaligned_padding(%unaligned_in) : (tensor<12xi32>) -> tensor<10xi32>
  %u_s0 = "stablehlo.slice"(%unaligned_in) {start_indices = array<i64: 0>, limit_indices = array<i64: 3>, strides = array<i64: 1>} : (tensor<12xi32>) -> tensor<3xi32>
  %u_s1 = "stablehlo.slice"(%unaligned_in) {start_indices = array<i64: 3>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<12xi32>) -> tensor<3xi32>
  %u_s2 = "stablehlo.slice"(%unaligned_in) {start_indices = array<i64: 6>, limit_indices = array<i64: 9>, strides = array<i64: 1>} : (tensor<12xi32>) -> tensor<3xi32>
  %u_s3 = "stablehlo.slice"(%unaligned_in) {start_indices = array<i64: 9>, limit_indices = array<i64: 12>, strides = array<i64: 1>} : (tensor<12xi32>) -> tensor<3xi32>
  %unaligned_res:4 = "interpreter.run_parallel"(%u_s0, %u_s1, %u_s2, %u_s3) {
    programs = [[@parallel_rotate_right_unaligned_padding, @parallel_rotate_right_unaligned_padding, @parallel_rotate_right_unaligned_padding, @parallel_rotate_right_unaligned_padding]]
  } : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<10xi32>, tensor<10xi32>, tensor<10xi32>, tensor<10xi32>)
  check.expect_eq %seq_unaligned, %unaligned_res#0 : tensor<10xi32>
  check.expect_eq %seq_unaligned, %unaligned_res#1 : tensor<10xi32>
  check.expect_eq %seq_unaligned, %unaligned_res#2 : tensor<10xi32>
  check.expect_eq %seq_unaligned, %unaligned_res#3 : tensor<10xi32>

  // Test 4: 2D rotate-right along dim = 1 on @mesh_2x2 (tensor<4x10xi32>).
  %rot_2d_iota = stablehlo.iota dim = 0 : tensor<40xi32>
  %rot_2d_in = stablehlo.reshape %rot_2d_iota : (tensor<40xi32>) -> tensor<4x10xi32>
  %seq_rot_2d = func.call @sequential_rotate_right_2d(%rot_2d_in) : (tensor<4x10xi32>) -> tensor<4x10xi32>
  %r2d_s0 = "stablehlo.slice"(%rot_2d_in) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 5>, strides = array<i64: 1, 1>} : (tensor<4x10xi32>) -> tensor<2x5xi32>
  %r2d_s1 = "stablehlo.slice"(%rot_2d_in) {start_indices = array<i64: 0, 5>, limit_indices = array<i64: 2, 10>, strides = array<i64: 1, 1>} : (tensor<4x10xi32>) -> tensor<2x5xi32>
  %r2d_s2 = "stablehlo.slice"(%rot_2d_in) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 5>, strides = array<i64: 1, 1>} : (tensor<4x10xi32>) -> tensor<2x5xi32>
  %r2d_s3 = "stablehlo.slice"(%rot_2d_in) {start_indices = array<i64: 2, 5>, limit_indices = array<i64: 4, 10>, strides = array<i64: 1, 1>} : (tensor<4x10xi32>) -> tensor<2x5xi32>
  %rot_2d_res:4 = "interpreter.run_parallel"(%r2d_s0, %r2d_s1, %r2d_s2, %r2d_s3) {
    programs = [[@parallel_rotate_right_2d, @parallel_rotate_right_2d, @parallel_rotate_right_2d, @parallel_rotate_right_2d]]
  } : (tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x5xi32>) ->
      (tensor<4x10xi32>, tensor<4x10xi32>, tensor<4x10xi32>, tensor<4x10xi32>)
  check.expect_eq %seq_rot_2d, %rot_2d_res#0 : tensor<4x10xi32>
  check.expect_eq %seq_rot_2d, %rot_2d_res#1 : tensor<4x10xi32>
  check.expect_eq %seq_rot_2d, %rot_2d_res#2 : tensor<4x10xi32>
  check.expect_eq %seq_rot_2d, %rot_2d_res#3 : tensor<4x10xi32>

  // Test 5: Rotate-right with both participatingShards < shardCount and rightPadding > 0
  // on @mesh_4 (tensor<5xi32>, participatingShards = 3, rightPadding = 1).
  %ps_in = stablehlo.constant dense<[0, 1, 2, 3, 4, 100, 200, 300]> : tensor<8xi32>
  %seq_ps = func.call @sequential_rotate_right_unaligned_partial_shards(%ps_in) : (tensor<8xi32>) -> tensor<5xi32>
  %ps_s0 = "stablehlo.slice"(%ps_in) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %ps_s1 = "stablehlo.slice"(%ps_in) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %ps_s2 = "stablehlo.slice"(%ps_in) {start_indices = array<i64: 4>, limit_indices = array<i64: 6>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %ps_s3 = "stablehlo.slice"(%ps_in) {start_indices = array<i64: 6>, limit_indices = array<i64: 8>, strides = array<i64: 1>} : (tensor<8xi32>) -> tensor<2xi32>
  %ps_res:4 = "interpreter.run_parallel"(%ps_s0, %ps_s1, %ps_s2, %ps_s3) {
    programs = [[@parallel_rotate_right_unaligned_partial_shards, @parallel_rotate_right_unaligned_partial_shards, @parallel_rotate_right_unaligned_partial_shards, @parallel_rotate_right_unaligned_partial_shards]]
  } : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<5xi32>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>)
  check.expect_eq %seq_ps, %ps_res#0 : tensor<5xi32>
  check.expect_eq %seq_ps, %ps_res#1 : tensor<5xi32>
  check.expect_eq %seq_ps, %ps_res#2 : tensor<5xi32>
  check.expect_eq %seq_ps, %ps_res#3 : tensor<5xi32>

  return
}
