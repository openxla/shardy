// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true --partition_count=4
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false --partition_count=4

//--- part1.mlir

sdy.mesh @mesh_x2_y2 = <["x"=2, "y"=2]>

// Case 1: 2D indivisible sharded inputs (5x7 sliced from divisible 6x8 on 2x2
// mesh -> padded back to 6x8, local 3x4 per shard) with mismatched operand
// shardings requiring explicit collective_permute, heterogeneous element types
// (f32, i1) -> i32, and all-gather + slice back to replicated output.
func.func @parallel_map_2d_indivisible_mismatched_io(
    %arg0: tensor<6x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{"x"}, {"y"}]>},
    %arg1: tensor<6x8xi1> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{"y"}, {"x"}]>})
    -> (tensor<5x7xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %s0 = stablehlo.slice %arg0 [0:5, 0:7] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"x"}, {"y"}]>]>} : (tensor<6x8xf32>) -> tensor<5x7xf32>
  %s1 = stablehlo.slice %arg1 [0:5, 0:7] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"y"}, {"x"}]>]>} : (tensor<6x8xi1>) -> tensor<5x7xi1>
  %0 = "stablehlo.map"(%s0, %s1) ({
    ^bb0(%v: tensor<f32>, %m: tensor<i1>):
      %scale = stablehlo.constant dense<10.0> : tensor<f32>
      %scaled = stablehlo.multiply %v, %scale : tensor<f32>
      %as_i32 = stablehlo.convert %scaled : (tensor<f32>) -> tensor<i32>
      %neg = stablehlo.negate %as_i32 : tensor<i32>
      %out = stablehlo.select %m, %as_i32, %neg : tensor<i1>, tensor<i32>
      stablehlo.return %out : tensor<i32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"x"}, {"y"}]>]>}
    : (tensor<5x7xf32>, tensor<5x7xi1>) -> tensor<5x7xi32>
  return %0 : tensor<5x7xi32>
}

func.func @sequential_map_2d_indivisible_mismatched_io(
    %arg0: tensor<6x8xf32>, %arg1: tensor<6x8xi1>) -> tensor<5x7xi32> {
  %s0 = stablehlo.slice %arg0 [0:5, 0:7] : (tensor<6x8xf32>) -> tensor<5x7xf32>
  %s1 = stablehlo.slice %arg1 [0:5, 0:7] : (tensor<6x8xi1>) -> tensor<5x7xi1>
  %0 = "stablehlo.map"(%s0, %s1) ({
    ^bb0(%v: tensor<f32>, %m: tensor<i1>):
      %scale = stablehlo.constant dense<10.0> : tensor<f32>
      %scaled = stablehlo.multiply %v, %scale : tensor<f32>
      %as_i32 = stablehlo.convert %scaled : (tensor<f32>) -> tensor<i32>
      %neg = stablehlo.negate %as_i32 : tensor<i32>
      %out = stablehlo.select %m, %as_i32, %neg : tensor<i1>, tensor<i32>
      stablehlo.return %out : tensor<i32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<5x7xf32>, tensor<5x7xi1>) -> tensor<5x7xi32>
  return %0 : tensor<5x7xi32>
}

// Case 2: Dim size smaller than shard count (3x5 across 4 shards -> padded 4x5,
// so shard 3 is 100% zero padding). The map body computes 1.0 / x + 5.0, which
// yields +Inf in the padded shard. Downstream reshard (all-gather + slice) and
// reduce must properly discard/mask out the +Inf padding slots.
func.func @parallel_map_empty_shard_inf_padding(
    %arg0: tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
    -> (tensor<3x5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
        tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}]>}) {
  %sharded = sdy.reshard %arg0 <@mesh_x2_y2, [{"x", "y"}, {}]> : tensor<3x5xf32>
  %mapped = "stablehlo.map"(%sharded) ({
    ^bb0(%x: tensor<f32>):
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %five = stablehlo.constant dense<5.0> : tensor<f32>
      %inv = stablehlo.divide %one, %x : tensor<f32>
      %out = stablehlo.add %inv, %five : tensor<f32>
      stablehlo.return %out : tensor<f32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"x", "y"}, {}]>]>}
    : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %gathered = sdy.reshard %mapped <@mesh_x2_y2, [{}, {}]> : tensor<3x5xf32>
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %reduced = stablehlo.reduce(%mapped init: %c0) applies stablehlo.add across dimensions = [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{}]>]>} : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>
  return %gathered, %reduced : tensor<3x5xf32>, tensor<5xf32>
}

func.func @sequential_map_empty_shard_inf_padding(
    %arg0: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<5xf32>) {
  %mapped = "stablehlo.map"(%arg0) ({
    ^bb0(%x: tensor<f32>):
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %five = stablehlo.constant dense<5.0> : tensor<f32>
      %inv = stablehlo.divide %one, %x : tensor<f32>
      %out = stablehlo.add %inv, %five : tensor<f32>
      stablehlo.return %out : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %reduced = stablehlo.reduce(%mapped init: %c0) applies stablehlo.add across dimensions = [0]
    : (tensor<3x5xf32>, tensor<f32>) -> tensor<5xf32>
  return %mapped, %reduced : tensor<3x5xf32>, tensor<5xf32>
}

// Case 3: 2D indivisible sharded map capturing an outer scalar bias with
// non-zero output in padded lanes along both dims, followed by full 2D reduce.
func.func @parallel_map_2d_indivisible_captured_and_reduce(
    %arg0: tensor<5x7xf32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
    %bias: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, []>})
    -> (tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, []>}) {
  %sharded = sdy.reshard %arg0 <@mesh_x2_y2, [{"x"}, {"y"}]> : tensor<5x7xf32>
  %mapped = "stablehlo.map"(%sharded) ({
    ^bb0(%x: tensor<f32>):
      %two = stablehlo.constant dense<2.0> : tensor<f32>
      %scaled = stablehlo.multiply %x, %two : tensor<f32>
      %biased = stablehlo.add %scaled, %bias : tensor<f32>
      stablehlo.return %biased : tensor<f32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"x"}, {"y"}]>]>}
    : (tensor<5x7xf32>) -> tensor<5x7xf32>
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %reduced = stablehlo.reduce(%mapped init: %c0) applies stablehlo.add across dimensions = [0, 1]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, []>]>} : (tensor<5x7xf32>, tensor<f32>) -> tensor<f32>
  return %reduced : tensor<f32>
}

func.func @sequential_map_2d_indivisible_captured_and_reduce(
    %arg0: tensor<5x7xf32>, %bias: tensor<f32>) -> tensor<f32> {
  %mapped = "stablehlo.map"(%arg0) ({
    ^bb0(%x: tensor<f32>):
      %two = stablehlo.constant dense<2.0> : tensor<f32>
      %scaled = stablehlo.multiply %x, %two : tensor<f32>
      %biased = stablehlo.add %scaled, %bias : tensor<f32>
      stablehlo.return %biased : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<5x7xf32>) -> tensor<5x7xf32>
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %reduced = stablehlo.reduce(%mapped init: %c0) applies stablehlo.add across dimensions = [0, 1]
    : (tensor<5x7xf32>, tensor<f32>) -> tensor<f32>
  return %reduced : tensor<f32>
}

//--- part2.mlir

func.func @main() {
  // === Case 1 ===
  %in0 = stablehlo.constant dense<[
    [ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0],
    [ 8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
    [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
    [ 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5],
    [ 2.0, -3.0,  4.0, -5.0,  6.0, -7.0,  8.0]
  ]> : tensor<5x7xf32>
  %in1 = stablehlo.constant dense<[
    [true,  false, true,  false, true,  false, true ],
    [false, true,  false, true,  false, true,  false],
    [true,  true,  false, false, true,  true,  false],
    [false, false, true,  true,  false, false, true ],
    [true,  false, false, true,  true,  false, true ]
  ]> : tensor<5x7xi1>

  // Pad inputs from 5x7 to 6x8 (with non-zero sentinel padding) to slice into
  // 4 divisible local 3x4 shards for mesh_x2_y2.
  %c999_f32 = stablehlo.constant dense<999.0> : tensor<f32>
  %ctrue = stablehlo.constant dense<true> : tensor<i1>
  %in0_pad = stablehlo.pad %in0, %c999_f32, low = [0, 0], high = [1, 1], interior = [0, 0]
    : (tensor<5x7xf32>, tensor<f32>) -> tensor<6x8xf32>
  %in1_pad = stablehlo.pad %in1, %ctrue, low = [0, 0], high = [1, 1], interior = [0, 0]
    : (tensor<5x7xi1>, tensor<i1>) -> tensor<6x8xi1>

  %seq1 = func.call @sequential_map_2d_indivisible_mismatched_io(%in0_pad, %in1_pad)
    : (tensor<6x8xf32>, tensor<6x8xi1>) -> tensor<5x7xi32>

  // %arg0 sharded [{"x"}, {"y"}]: device (x,y) = (0,0),(0,1),(1,0),(1,1)
  %a0_00 = stablehlo.slice %in0_pad [0:3, 0:4] : (tensor<6x8xf32>) -> tensor<3x4xf32>
  %a0_01 = stablehlo.slice %in0_pad [0:3, 4:8] : (tensor<6x8xf32>) -> tensor<3x4xf32>
  %a0_10 = stablehlo.slice %in0_pad [3:6, 0:4] : (tensor<6x8xf32>) -> tensor<3x4xf32>
  %a0_11 = stablehlo.slice %in0_pad [3:6, 4:8] : (tensor<6x8xf32>) -> tensor<3x4xf32>

  // %arg1 sharded [{"y"}, {"x"}]: device (0,0)->(0,0), (0,1)->(1,0), (1,0)->(0,1), (1,1)->(1,1)
  %a1_00 = stablehlo.slice %in1_pad [0:3, 0:4] : (tensor<6x8xi1>) -> tensor<3x4xi1>
  %a1_01 = stablehlo.slice %in1_pad [3:6, 0:4] : (tensor<6x8xi1>) -> tensor<3x4xi1>
  %a1_10 = stablehlo.slice %in1_pad [0:3, 4:8] : (tensor<6x8xi1>) -> tensor<3x4xi1>
  %a1_11 = stablehlo.slice %in1_pad [3:6, 4:8] : (tensor<6x8xi1>) -> tensor<3x4xi1>

  %res1:4 = "interpreter.run_parallel"(
      %a0_00, %a1_00, %a0_01, %a1_01, %a0_10, %a1_10, %a0_11, %a1_11) {
    programs = [[@parallel_map_2d_indivisible_mismatched_io,
                 @parallel_map_2d_indivisible_mismatched_io,
                 @parallel_map_2d_indivisible_mismatched_io,
                 @parallel_map_2d_indivisible_mismatched_io]]
  } : (tensor<3x4xf32>, tensor<3x4xi1>, tensor<3x4xf32>, tensor<3x4xi1>,
       tensor<3x4xf32>, tensor<3x4xi1>, tensor<3x4xf32>, tensor<3x4xi1>)
    -> (tensor<5x7xi32>, tensor<5x7xi32>, tensor<5x7xi32>, tensor<5x7xi32>)

  "check.expect_eq"(%res1#0, %seq1) : (tensor<5x7xi32>, tensor<5x7xi32>) -> ()
  "check.expect_eq"(%res1#1, %seq1) : (tensor<5x7xi32>, tensor<5x7xi32>) -> ()
  "check.expect_eq"(%res1#2, %seq1) : (tensor<5x7xi32>, tensor<5x7xi32>) -> ()
  "check.expect_eq"(%res1#3, %seq1) : (tensor<5x7xi32>, tensor<5x7xi32>) -> ()

  // === Case 2 ===
  %in2 = stablehlo.constant dense<[
    [1.0, 2.0, 4.0, 5.0, 8.0],
    [0.5, 1.0, 2.0, 4.0, 10.0],
    [2.0, 4.0, 5.0, 8.0, 20.0]
  ]> : tensor<3x5xf32>
  %seq2:2 = func.call @sequential_map_empty_shard_inf_padding(%in2)
    : (tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<5xf32>)

  %res2:8 = "interpreter.run_parallel"(%in2, %in2, %in2, %in2) {
    programs = [[@parallel_map_empty_shard_inf_padding,
                 @parallel_map_empty_shard_inf_padding,
                 @parallel_map_empty_shard_inf_padding,
                 @parallel_map_empty_shard_inf_padding]]
  } : (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
    -> (tensor<3x5xf32>, tensor<5xf32>, tensor<3x5xf32>, tensor<5xf32>,
        tensor<3x5xf32>, tensor<5xf32>, tensor<3x5xf32>, tensor<5xf32>)

  "check.expect_eq"(%res2#0, %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#2, %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#4, %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#6, %seq2#0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
  "check.expect_eq"(%res2#1, %seq2#1) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#3, %seq2#1) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#5, %seq2#1) : (tensor<5xf32>, tensor<5xf32>) -> ()
  "check.expect_eq"(%res2#7, %seq2#1) : (tensor<5xf32>, tensor<5xf32>) -> ()

  // === Case 3 ===
  %bias = stablehlo.constant dense<-3.5> : tensor<f32>
  %seq3 = func.call @sequential_map_2d_indivisible_captured_and_reduce(%in0, %bias)
    : (tensor<5x7xf32>, tensor<f32>) -> tensor<f32>

  %res3:4 = "interpreter.run_parallel"(%in0, %bias, %in0, %bias, %in0, %bias, %in0, %bias) {
    programs = [[@parallel_map_2d_indivisible_captured_and_reduce,
                 @parallel_map_2d_indivisible_captured_and_reduce,
                 @parallel_map_2d_indivisible_captured_and_reduce,
                 @parallel_map_2d_indivisible_captured_and_reduce]]
  } : (tensor<5x7xf32>, tensor<f32>, tensor<5x7xf32>, tensor<f32>,
       tensor<5x7xf32>, tensor<f32>, tensor<5x7xf32>, tensor<f32>)
    -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)

  "check.expect_eq"(%res3#0, %seq3) : (tensor<f32>, tensor<f32>) -> ()
  "check.expect_eq"(%res3#1, %seq3) : (tensor<f32>, tensor<f32>) -> ()
  "check.expect_eq"(%res3#2, %seq3) : (tensor<f32>, tensor<f32>) -> ()
  "check.expect_eq"(%res3#3, %seq3) : (tensor<f32>, tensor<f32>) -> ()

  return
}
