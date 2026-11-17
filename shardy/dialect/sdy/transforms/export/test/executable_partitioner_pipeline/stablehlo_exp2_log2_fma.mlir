// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true --partition_count=4
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false --partition_count=4

//--- part1.mlir

sdy.mesh @mesh_x2_y2 = <["x"=2, "y"=2]>

// Case 1: Divisible 2D sharded exp2 (with non-default result_accuracy), log2,
// and 3-operand fma with mismatched operand and result shardings requiring
// explicit reshards across the 2x2 mesh.
func.func @parallel_exp2_log2_fma_divisible(
    %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{"x"}, {"y"}]>},
    %arg1: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{"y"}, {"x"}]>},
    %arg2: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {"y"}]>})
    -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{"y"}, {"x"}]>},
        tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{"x"}, {}]>},
        tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{"x"}, {"y"}]>}) {
  %exp2 = stablehlo.exp2 %arg0 {
    result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"y"}, {"x"}]>]>
  } : tensor<4x4xf32>
  %log2 = stablehlo.log2 %arg1 {
    result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {}]>]>
  } : tensor<4x4xf32>
  %fma = stablehlo.fma %arg0, %arg1, %arg2 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {"y"}]>]>
  } : tensor<4x4xf32>
  return %exp2, %log2, %fma : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

func.func @sequential_exp2_log2_fma_divisible(
    %arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %exp2 = stablehlo.exp2 %arg0 {
    result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : tensor<4x4xf32>
  %log2 = stablehlo.log2 %arg1 {
    result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : tensor<4x4xf32>
  %fma = stablehlo.fma %arg0, %arg1, %arg2 : tensor<4x4xf32>
  return %exp2, %log2, %fma : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

// Case 2: Indivisible sharding where dim size < shard count (3x5 across 4
// shards -> padded to 4x5, so shard 3 is 100% zero padding). Zero padding
// produces non-zero exp2(0) = 1.0, log2(0) = -inf, and fma(0, 0, 1.0) = 1.0.
// Downstream all-gather + slice and reduce must discard and mask those padded
// lanes rather than letting 1.0 or -inf leak into results.
func.func @parallel_exp2_log2_fma_indivisible_empty_shard(
    %arg0: tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg1: tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
    -> (tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
        tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
        tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
        tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>},
        tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>},
        tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>}) {
  %s0 = sdy.reshard %arg0 <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<3x5xf32>
  %s1 = sdy.reshard %arg1 <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<3x5xf32>
  %exp2 = stablehlo.exp2 %s0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x", "y"}, {}]>]>} : tensor<3x5xf32>
  %log2 = stablehlo.log2 %s1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x", "y"}, {}]>]>} : tensor<3x5xf32>
  %fma = stablehlo.fma %s0, %s1, %exp2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x", "y"}, {}]>]>} : tensor<3x5xf32>

  %exp2_g = sdy.reshard %exp2 <@mesh_x2_y2, [{}, {}]> : tensor<3x5xf32>
  %log2_g = sdy.reshard %log2 <@mesh_x2_y2, [{}, {}]> : tensor<3x5xf32>
  %fma_g = sdy.reshard %fma <@mesh_x2_y2, [{}, {}]> : tensor<3x5xf32>

  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %exp2_r = stablehlo.reduce(%exp2 init: %c0) applies stablehlo.add across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{}]>]>} : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>
  %log2_r = stablehlo.reduce(%log2 init: %c0) applies stablehlo.add across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{}]>]>} : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>
  %fma_r = stablehlo.reduce(%fma init: %c0) applies stablehlo.add across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{}]>]>} : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>

  return %exp2_g, %log2_g, %fma_g, %exp2_r, %log2_r, %fma_r
    : tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>,
      tensor<5xf32>, tensor<5xf32>, tensor<5xf32>
}

func.func @sequential_exp2_log2_fma_indivisible_empty_shard(
    %arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
    -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>,
        tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) {
  %exp2 = stablehlo.exp2 %arg0 : tensor<3x5xf32>
  %log2 = stablehlo.log2 %arg1 : tensor<3x5xf32>
  %fma = stablehlo.fma %arg0, %arg1, %exp2 : tensor<3x5xf32>
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %exp2_r = stablehlo.reduce(%exp2 init: %c0) applies stablehlo.add across dimensions = [0]
    : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>
  %log2_r = stablehlo.reduce(%log2 init: %c0) applies stablehlo.add across dimensions = [0]
    : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>
  %fma_r = stablehlo.reduce(%fma init: %c0) applies stablehlo.add across dimensions = [0]
    : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>
  return %exp2, %log2, %fma, %exp2_r, %log2_r, %fma_r
    : tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>,
      tensor<5xf32>, tensor<5xf32>, tensor<5xf32>
}

// Case 3: 2D indivisible sharded inputs (5x7 on 2x2 mesh -> padded to 6x8,
// local 3x4 per shard) with mismatched operand shardings feeding exp2, log2,
// and fma, followed by a full 2D reduction.
func.func @parallel_exp2_log2_fma_2d_indivisible_mismatched(
    %arg0: tensor<5x7xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %arg1: tensor<5x7xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
    -> (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, []>}) {
  %s0 = sdy.reshard %arg0 <@mesh_x2_y2, [{"x"}, {"y"}]> : tensor<5x7xf32>
  %s1 = sdy.reshard %arg1 <@mesh_x2_y2, [{"y"}, {"x"}]> : tensor<5x7xf32>
  %exp2 = stablehlo.exp2 %s0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {"y"}]>]>} : tensor<5x7xf32>
  %log2 = stablehlo.log2 %s1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"y"}, {"x"}]>]>} : tensor<5x7xf32>
  %fma = stablehlo.fma %exp2, %log2, %s0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {"y"}]>]>} : tensor<5x7xf32>
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %reduced = stablehlo.reduce(%fma init: %c0) applies stablehlo.add across dimensions = [0, 1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, []>]>} : (tensor<5x7xf32>, tensor<f32>) -> tensor<f32>
  return %reduced : tensor<f32>
}

func.func @sequential_exp2_log2_fma_2d_indivisible_mismatched(
    %arg0: tensor<5x7xf32>, %arg1: tensor<5x7xf32>) -> tensor<f32> {
  %exp2 = stablehlo.exp2 %arg0 : tensor<5x7xf32>
  %log2 = stablehlo.log2 %arg1 : tensor<5x7xf32>
  %fma = stablehlo.fma %exp2, %log2, %arg0 : tensor<5x7xf32>
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %reduced = stablehlo.reduce(%fma init: %c0) applies stablehlo.add across dimensions = [0, 1]
    : (tensor<5x7xf32>, tensor<f32>) -> tensor<f32>
  return %reduced : tensor<f32>
}

//--- part2.mlir

func.func @main() {
  // === Case 1: Divisible 2D sharded ===
  // Includes non-integer exponents and a single-rounding fma element.
  %d0 = stablehlo.constant dense<[
    [-2.0, -1.0,  0.0,  1.0],
    [ 2.0,  3.0,  0.5, -0.5],
    [ 1.5, -1.5,  2.5,  0.0],
    [ 1.00000012, -2.0, 3.0, 4.0]
  ]> : tensor<4x4xf32>
  %d1 = stablehlo.constant dense<[
    [ 0.25,  0.5,  1.0,  2.0],
    [ 4.0,   8.0, 16.0,  3.0],
    [ 5.0,   6.0,  7.0,  9.0],
    [ 0.99999994, 0.5, 2.0, 0.25]
  ]> : tensor<4x4xf32>
  %d2 = stablehlo.constant dense<[
    [ 1.0,  -0.5,  2.25, -3.5],
    [ 0.75,  4.0, -1.25,  2.0],
    [-2.0,   3.0,  0.5,  -4.0],
    [-1.0,   1.5, -2.5,   3.25]
  ]> : tensor<4x4xf32>

  %seq1:3 = func.call @sequential_exp2_log2_fma_divisible(%d0, %d1, %d2)
    : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)

  // %arg0 sharded [{"x"}, {"y"}]: (0,0),(0,1),(1,0),(1,1)
  %d0_00 = stablehlo.slice %d0 [0:2, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %d0_01 = stablehlo.slice %d0 [0:2, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %d0_10 = stablehlo.slice %d0 [2:4, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %d0_11 = stablehlo.slice %d0 [2:4, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>

  // %arg1 sharded [{"y"}, {"x"}]: dev(0,0)->(0,0), dev(0,1)->(1,0), dev(1,0)->(0,1), dev(1,1)->(1,1)
  %d1_00 = stablehlo.slice %d1 [0:2, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %d1_01 = stablehlo.slice %d1 [2:4, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %d1_10 = stablehlo.slice %d1 [0:2, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>
  %d1_11 = stablehlo.slice %d1 [2:4, 2:4] : (tensor<4x4xf32>) -> tensor<2x2xf32>

  // %arg2 sharded [{}, {"y"}]: dev(0,0)->col 0:2, dev(0,1)->col 2:4, dev(1,0)->col 0:2, dev(1,1)->col 2:4
  %d2_c0 = stablehlo.slice %d2 [0:4, 0:2] : (tensor<4x4xf32>) -> tensor<4x2xf32>
  %d2_c1 = stablehlo.slice %d2 [0:4, 2:4] : (tensor<4x4xf32>) -> tensor<4x2xf32>

  %res1:12 = "interpreter.run_parallel"(
      %d0_00, %d1_00, %d2_c0,
      %d0_01, %d1_01, %d2_c1,
      %d0_10, %d1_10, %d2_c0,
      %d0_11, %d1_11, %d2_c1) {
    programs = [[@parallel_exp2_log2_fma_divisible,
                 @parallel_exp2_log2_fma_divisible,
                 @parallel_exp2_log2_fma_divisible,
                 @parallel_exp2_log2_fma_divisible]]
  } : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>,
       tensor<2x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>,
       tensor<2x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>,
       tensor<2x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>)
    -> (tensor<2x2xf32>, tensor<2x4xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x4xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x4xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x4xf32>, tensor<2x2xf32>)

  // Reconstruct exp2 output (sharded [{"y"}, {"x"}])
  %exp2_r0 = stablehlo.concatenate %res1#0, %res1#6, dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %exp2_r1 = stablehlo.concatenate %res1#3, %res1#9, dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %exp2_full = stablehlo.concatenate %exp2_r0, %exp2_r1, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  "check.expect_eq"(%exp2_full, %seq1#0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  // Reconstruct log2 output (sharded [{"x"}, {}]): dev(0,0)==dev(0,1), dev(1,0)==dev(1,1)
  %log2_full_0 = stablehlo.concatenate %res1#1, %res1#7, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  %log2_full_1 = stablehlo.concatenate %res1#4, %res1#10, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  "check.expect_eq"(%log2_full_0, %seq1#1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()
  "check.expect_eq"(%log2_full_1, %seq1#1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  // Reconstruct fma output (sharded [{"x"}, {"y"}])
  %fma_r0 = stablehlo.concatenate %res1#2, %res1#5, dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %fma_r1 = stablehlo.concatenate %res1#8, %res1#11, dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  %fma_full = stablehlo.concatenate %fma_r0, %fma_r1, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  "check.expect_eq"(%fma_full, %seq1#2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  // === Case 2: Indivisible with 100%-padded empty shard ===
  // Exact integer exponents and power-of-2 log2 arguments so all results and
  // reductions are exact in IEEE-754 f32.
  %i0 = stablehlo.constant dense<[
    [-2.0, -1.0, 0.0, 1.0, 2.0],
    [ 3.0,  2.0, 1.0, 0.0, -1.0],
    [ 0.0,  1.0, 2.0, 3.0, -2.0]
  ]> : tensor<3x5xf32>
  %i1 = stablehlo.constant dense<[
    [0.25, 0.5, 1.0, 2.0, 4.0],
    [8.0,  4.0, 2.0, 1.0, 0.5],
    [1.0,  2.0, 4.0, 8.0, 16.0]
  ]> : tensor<3x5xf32>

  %seq2:6 = func.call @sequential_exp2_log2_fma_indivisible_empty_shard(%i0, %i1)
    : (tensor<3x5xf32>, tensor<3x5xf32>)
    -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>,
        tensor<5xf32>, tensor<5xf32>, tensor<5xf32>)

  %res2:24 = "interpreter.run_parallel"(%i0, %i1, %i0, %i1, %i0, %i1, %i0, %i1) {
    programs = [[@parallel_exp2_log2_fma_indivisible_empty_shard,
                 @parallel_exp2_log2_fma_indivisible_empty_shard,
                 @parallel_exp2_log2_fma_indivisible_empty_shard,
                 @parallel_exp2_log2_fma_indivisible_empty_shard]]
  } : (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>,
       tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
    -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>,
        tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>,
        tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>,
        tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>)

  "check.expect_eq"(%res2#0,  %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#6,  %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#12, %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#18, %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()

  "check.expect_eq"(%res2#1,  %seq2#1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#7,  %seq2#1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#13, %seq2#1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#19, %seq2#1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()

  "check.expect_eq"(%res2#2,  %seq2#2) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#8,  %seq2#2) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#14, %seq2#2) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#20, %seq2#2) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()

  "check.expect_eq"(%res2#3,  %seq2#3) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#9,  %seq2#3) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#15, %seq2#3) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#21, %seq2#3) : (tensor<5xf32>, tensor<5xf32>) -> ()

  "check.expect_eq"(%res2#4,  %seq2#4) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#10, %seq2#4) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#16, %seq2#4) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#22, %seq2#4) : (tensor<5xf32>, tensor<5xf32>) -> ()

  "check.expect_eq"(%res2#5,  %seq2#5) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#11, %seq2#5) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#17, %seq2#5) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#23, %seq2#5) : (tensor<5xf32>, tensor<5xf32>) -> ()

  // === Case 3: 2D indivisible with mismatched operand shardings ===
  %m0 = stablehlo.constant dense<[
    [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -1.0],
    [ 1.0,  0.0, 2.0, -2.0, 1.0, 0.0,  2.0],
    [ 2.0,  3.0, -1.0, 0.0, -2.0, 1.0, 0.0],
    [-1.0,  1.0, 0.0, 2.0, 3.0, -1.0, 1.0],
    [ 0.0, -2.0, 1.0, 3.0, 0.0, 2.0, -2.0]
  ]> : tensor<5x7xf32>
  %m1 = stablehlo.constant dense<[
    [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25],
    [1.0,  2.0, 0.5, 4.0, 0.25, 8.0, 2.0],
    [4.0,  1.0, 8.0, 0.5, 2.0, 16.0, 0.25],
    [2.0,  4.0, 1.0, 8.0, 0.5, 0.25, 1.0]
  ]> : tensor<5x7xf32>

  %seq3 = func.call @sequential_exp2_log2_fma_2d_indivisible_mismatched(%m0, %m1)
    : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<f32>

  %res3:4 = "interpreter.run_parallel"(%m0, %m1, %m0, %m1, %m0, %m1, %m0, %m1) {
    programs = [[@parallel_exp2_log2_fma_2d_indivisible_mismatched,
                 @parallel_exp2_log2_fma_2d_indivisible_mismatched,
                 @parallel_exp2_log2_fma_2d_indivisible_mismatched,
                 @parallel_exp2_log2_fma_2d_indivisible_mismatched]]
  } : (tensor<5x7xf32>, tensor<5x7xf32>, tensor<5x7xf32>, tensor<5x7xf32>,
       tensor<5x7xf32>, tensor<5x7xf32>, tensor<5x7xf32>, tensor<5x7xf32>)
    -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)

  "check.expect_eq"(%res3#0, %seq3) : (tensor<f32>, tensor<f32>) -> ()
  "check.expect_eq"(%res3#1, %seq3) : (tensor<f32>, tensor<f32>) -> ()
  "check.expect_eq"(%res3#2, %seq3) : (tensor<f32>, tensor<f32>) -> ()
  "check.expect_eq"(%res3#3, %seq3) : (tensor<f32>, tensor<f32>) -> ()

  return
}
