// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"

// Resharding pattern:
// - Mesh: 4 devices @mesh = <["x"=2, "y"=2]>.
// - Resharding: [{"x", "y"}, {}] -> [{"y"}, {"x"}].
//
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
// dim 0) followed by sdy.all_to_all (scattering "x" to dim 1).
// OptimizeCollectivesPass detects the collective_permute + all_to_all chain and
// eliminates the collective_permute by decomposing dim 0 shape and issuing a
// single multi-axis all_to_all.
// The permuted axes ("x" and "y") are partially scattered off dimension 0 (only
// "x" is scattered to dimension 1, while "y" remains on dimension 0).
//
// Numerically verified via interpreter.run_parallel.

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>
sdy.mesh @mesh3d = <["a"=2, "b"=2, "d"=2]>

func.func @parallel_sdy_all_to_all_partially_scattered(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {"x"}]> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @sequential_sdy_all_to_all_partially_scattered(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  return %arg0 : tensor<4x4xf32>
}

func.func @parallel_sdy_reshard_uneven_all_slice_all_to_all(%arg0: tensor<5x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<5x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %0:2 = stablehlo.while(%iterArg = %c0, %iterArg_0 = %arg0) : tensor<i32>, tensor<5x4xf32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, []>, <@mesh, [{"x", "y"}, {}]>]>}
  cond {
    %cmp = stablehlo.compare LT, %iterArg, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next = stablehlo.add %iterArg, %c1 : tensor<i32>
    %add = stablehlo.add %iterArg_0, %iterArg_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} : tensor<5x4xf32>
    stablehlo.return %next, %add : tensor<i32>, tensor<5x4xf32>
  }
  return %0#1 : tensor<5x4xf32>
}

func.func @sequential_sdy_reshard_uneven_all_slice_all_to_all(%arg0: tensor<5x4xf32>) -> tensor<5x4xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

func.func @parallel_sdy_reshard_uneven_stage3_available_out_axes(%arg0: tensor<4x5xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"b", "a"}, {}]>}) -> (tensor<4x5xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{}, {}]>}) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %0:2 = stablehlo.while(%iterArg = %c0, %iterArg_0 = %arg0) : tensor<i32>, tensor<4x5xf32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, []>, <@mesh3d, [{"d"}, {"a", "b"}]>]>}
  cond {
    %cmp = stablehlo.compare LT, %iterArg, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  } do {
    %next = stablehlo.add %iterArg, %c1 : tensor<i32>
    %add = stablehlo.add %iterArg_0, %iterArg_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"d"}, {"a", "b"}]>]>} : tensor<4x5xf32>
    stablehlo.return %next, %add : tensor<i32>, tensor<4x5xf32>
  }
  return %0#1 : tensor<4x5xf32>
}

func.func @sequential_sdy_reshard_uneven_stage3_available_out_axes(%arg0: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

//--- part2.mlir
func.func @main() {
  %input = stablehlo.constant dense<[
    [ 0.0,  1.0,  2.0,  3.0],
    [ 4.0,  5.0,  6.0,  7.0],
    [ 8.0,  9.0, 10.0, 11.0],
    [12.0, 13.0, 14.0, 15.0]
  ]> : tensor<4x4xf32>
  %seq = func.call @sequential_sdy_all_to_all_partially_scattered(%input) : (tensor<4x4xf32>) -> tensor<4x4xf32>

  // Input slices (dim 0 sharded by {"x", "y"} across 4 devices):
  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<1x4xf32>

  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[
      @parallel_sdy_all_to_all_partially_scattered, @parallel_sdy_all_to_all_partially_scattered,
      @parallel_sdy_all_to_all_partially_scattered, @parallel_sdy_all_to_all_partially_scattered
    ]]
  } : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) ->
      (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)

  // Expected output slices (dim 0 sharded by "y", dim 1 sharded by "x"):
  // Device 0 (x=0, y=0)
  %exp0 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  // Device 1 (x=0, y=1)
  %exp1 = "stablehlo.slice"(%seq) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  // Device 2 (x=1, y=0)
  %exp2 = "stablehlo.slice"(%seq) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
  // Device 3 (x=1, y=1)
  %exp3 = "stablehlo.slice"(%seq) {start_indices = array<i64: 2, 2>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x2xf32>

  "check.expect_eq"(%res#0, %exp0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%res#1, %exp1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%res#2, %exp2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
  "check.expect_eq"(%res#3, %exp3) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()

  %input_uneven = stablehlo.constant dense<[
    [ 0.0,  1.0,  2.0,  3.0],
    [ 4.0,  5.0,  6.0,  7.0],
    [ 8.0,  9.0, 10.0, 11.0],
    [12.0, 13.0, 14.0, 15.0],
    [16.0, 17.0, 18.0, 19.0]
  ]> : tensor<5x4xf32>
  %seq_uneven = func.call @sequential_sdy_reshard_uneven_all_slice_all_to_all(%input_uneven) : (tensor<5x4xf32>) -> tensor<5x4xf32>

  // Input slices (dim 0 unsharded, dim 1 sharded by "y"):
  // Device 0 (x=0, y=0) and Device 2 (x=1, y=0) hold cols [0:2]
  %u_col0 = "stablehlo.slice"(%input_uneven) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 5, 2>, strides = array<i64: 1, 1>} : (tensor<5x4xf32>) -> tensor<5x2xf32>
  // Device 1 (x=0, y=1) and Device 3 (x=1, y=1) hold cols [2:4]
  %u_col1 = "stablehlo.slice"(%input_uneven) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 5, 4>, strides = array<i64: 1, 1>} : (tensor<5x4xf32>) -> tensor<5x2xf32>

  %res_u:4 = "interpreter.run_parallel"(%u_col0, %u_col1, %u_col0, %u_col1) {
    programs = [[
      @parallel_sdy_reshard_uneven_all_slice_all_to_all, @parallel_sdy_reshard_uneven_all_slice_all_to_all,
      @parallel_sdy_reshard_uneven_all_slice_all_to_all, @parallel_sdy_reshard_uneven_all_slice_all_to_all
    ]]
  } : (tensor<5x2xf32>, tensor<5x2xf32>, tensor<5x2xf32>, tensor<5x2xf32>) ->
      (tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>)

  "check.expect_eq"(%res_u#0, %seq_uneven) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  "check.expect_eq"(%res_u#1, %seq_uneven) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  "check.expect_eq"(%res_u#2, %seq_uneven) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  "check.expect_eq"(%res_u#3, %seq_uneven) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()

  %input_s3 = stablehlo.constant dense<[
    [ 0.0,  1.0,  2.0,  3.0,  4.0],
    [ 5.0,  6.0,  7.0,  8.0,  9.0],
    [10.0, 11.0, 12.0, 13.0, 14.0],
    [15.0, 16.0, 17.0, 18.0, 19.0]
  ]> : tensor<4x5xf32>
  %seq_s3 = func.call @sequential_sdy_reshard_uneven_stage3_available_out_axes(%input_s3) : (tensor<4x5xf32>) -> tensor<4x5xf32>

  // Input slices on @mesh3d <["a"=2, "b"=2, "d"=2]> with sharding [{"b", "a"}, {}]:
  // ("b"=0, "a"=0) -> devices 0, 1 hold row 0
  // ("b"=0, "a"=1) -> devices 4, 5 hold row 1
  // ("b"=1, "a"=0) -> devices 2, 3 hold row 2
  // ("b"=1, "a"=1) -> devices 6, 7 hold row 3
  %r0 = "stablehlo.slice"(%input_s3) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 5>, strides = array<i64: 1, 1>} : (tensor<4x5xf32>) -> tensor<1x5xf32>
  %r1 = "stablehlo.slice"(%input_s3) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 5>, strides = array<i64: 1, 1>} : (tensor<4x5xf32>) -> tensor<1x5xf32>
  %r2 = "stablehlo.slice"(%input_s3) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 5>, strides = array<i64: 1, 1>} : (tensor<4x5xf32>) -> tensor<1x5xf32>
  %r3 = "stablehlo.slice"(%input_s3) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 5>, strides = array<i64: 1, 1>} : (tensor<4x5xf32>) -> tensor<1x5xf32>

  %res_s3:8 = "interpreter.run_parallel"(%r0, %r0, %r2, %r2, %r1, %r1, %r3, %r3) {
    programs = [[
      @parallel_sdy_reshard_uneven_stage3_available_out_axes, @parallel_sdy_reshard_uneven_stage3_available_out_axes,
      @parallel_sdy_reshard_uneven_stage3_available_out_axes, @parallel_sdy_reshard_uneven_stage3_available_out_axes,
      @parallel_sdy_reshard_uneven_stage3_available_out_axes, @parallel_sdy_reshard_uneven_stage3_available_out_axes,
      @parallel_sdy_reshard_uneven_stage3_available_out_axes, @parallel_sdy_reshard_uneven_stage3_available_out_axes
    ]]
  } : (tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>) ->
      (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)

  "check.expect_eq"(%res_s3#0, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "check.expect_eq"(%res_s3#1, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "check.expect_eq"(%res_s3#2, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "check.expect_eq"(%res_s3#3, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "check.expect_eq"(%res_s3#4, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "check.expect_eq"(%res_s3#5, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "check.expect_eq"(%res_s3#6, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  "check.expect_eq"(%res_s3#7, %seq_s3) : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
  return
}
