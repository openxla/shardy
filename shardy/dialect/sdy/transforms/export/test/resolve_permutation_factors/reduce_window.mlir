// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>
// CHECK: @mesh_a6 = <["a"=6]>
sdy.mesh @mesh_a6 = <["a"=6]>
// CHECK: @mesh_a_4 = <["a"=4]>
sdy.mesh @mesh_a_4 = <["a"=4]>

// 1D window (3x1) with stride 1 and zero padding on tensor<8x8xi32> ->
// tensor<6x8xi32>, sharded 2-way on dim 0 along sub-axis "a":(2)2 (sIn = 4,
// sOut = 3) and 2-way on dim 1 along "b" (sIn = 4, sOut = 4, passthrough):
// - Replication mode: Replicates dim 0 before reduce_window and reshards the output.
// - HALO mode: Dim 0 has sFootprint = (3 - 1) * 1 + 3 = 5. Shard 0 needs global
//   elements [0..4] (rightHops = 1) and shard 1 needs [3..7] (leftHops = 1).
//   Exchanges 1 left and 1 right sliced halo neighbor along "a":(2)2, slices a
//   5x4 footprint at offset max(id * (-1) + 1, 0), and runs local reduce_window
//   producing tensor<3x4xi32>.
// CHECK-LABEL: func @reduce_window_1d_sharded_no_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_1d_sharded_no_pad(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>}) -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh_a4, [{"a":(2)2}, {"b"}]> : tensor<6x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>] manual_axes={"a":(2)2, "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM1:.*]] = stablehlo.remainder %[[DIV1]], %[[C_MOD1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[REM1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [3:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]> : tensor<4x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg1 [0:1, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [6, 4], [7, 5]]> : tensor<4x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[CONV1]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[MAX1]], %[[C_ZERO1]], sizes = [5, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE1]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<3x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
    }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x8xi32>

  // CHECK: return %[[RES]] : tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// 1D window (3x1) with stride [2, 1] and symmetric padding [[1, 1], [0, 0]] on
// tensor<8x8xi32> -> tensor<4x8xi32>, sharded 2-way on dim 0 (sIn = 4,
// sOut = 2) and 2-way on dim 1 (passthrough):
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: Dim 0 has sFootprint = (2 - 1) * 2 + 3 = 5. Shard 0 needs global
//   elements [-1..3] (within shard 0 plus low padding) and shard 1 needs [3..7]
//   (element 3 from shard 0, elements [4..7] from shard 1), so leftHops = 1 and
//   rightHops = 0. Concatenates [left, self], pads low by 1, slices a 5x4 footprint
//   at offset max(id * 0 + 1, 0) (diffSize = sOut * stride - sIn = 0), and runs
//   local reduce_window with zero local padding producing tensor<2x4xi32>.
// CHECK-LABEL: func @reduce_window_1d_sharded_padding_and_stride
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_1d_sharded_padding_and_stride(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 2, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<4x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [3:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [1, 0], high = [0, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[C_OFFSET1]], %[[C_ZERO1]], sizes = [5, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE1]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 2, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<2x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>

  // CHECK: return %[[RES]] : tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// 1D window (3x1) with window_dilations = [2, 1], stride 1, and zero padding on
// tensor<8x8xi32> -> tensor<4x8xi32>, sharded 2-way on dim 0 (sIn = 4,
// sOut = 2):
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: Dilated window size on dim 0 is 1 + (3 - 1) * 2 = 5, so
//   sFootprint = (2 - 1) * 1 + 5 = 6. Shard 0 needs [0..5] (rightHops = 1) and
//   shard 1 needs [2..7] (leftHops = 1). Exchanges 2-element left and right
//   sliced halos, slices a 6x4 footprint at offset max(id * (-2) + 2, 0), and
//   runs local reduce_window producing tensor<2x4xi32>.
// CHECK-LABEL: func @reduce_window_1d_window_dilation_only
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_1d_window_dilation_only(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{window_dilations = array<i64: 2, 1>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<4x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [2:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg1 [0:2, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[CONV1]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[MAX1]], %[[C_ZERO1]], sizes = [6, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE1]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 2, 1>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<2x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    window_dilations = array<i64: 2, 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>

  // CHECK: return %[[RES]] : tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// 1D window (3x1) with base_dilations = [3, 1], window_dilations = [2, 1],
// stride 1, and padding [[4, 6], [0, 0]] on tensor<4x8xi32> ->
// tensor<16x8xi32>, sharded 2-way on dim 0 (sIn = 2, sOut = 8):
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: On dim 0, sOut * stride = 8 is not divisible by baseDilation = 3, and
//   stride == 1. Across shards t in {0, 1}, maxRem = max((-4) mod 3, 4 mod 3)
//   = 2, dilatedSize = 5, sFootprintDilated = 12, and physical footprint
//   sFootprint = ceil((12 + 2 - 1) / 3) + 1 = 6 (leftHops = 1, rightHops = 1).
//   Slices a 6x4 input footprint at undilated offset dilatedOffset / 3
//   (dilatedOffset = max(id * 2 + 5, 0)), runs local reduce_window producing
//   inferredOutDim = 12 rows (tensor<12x4xi32>), and dynamically slices 8 rows
//   at post-op offset dilatedOffset % 3 to produce tensor<8x4xi32>.
// CHECK-LABEL: func @reduce_window_1d_base_and_window_dilations_low_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xi32>
func.func @reduce_window_1d_base_and_window_dilations_low_pad(%arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<4x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{base_dilations = array<i64: 3, 1>, padding = dense<[[4, 6], [0, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 2, 1>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<16x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<2x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [1:2, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg1 [0:2, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [2, 0], high = [3, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[CONV1]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[C_BD1:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[UNDIL1:.*]] = stablehlo.divide %[[MAX1]], %[[C_BD1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[UNDIL1]], %[[C_ZERO1]], sizes = [6, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE1]], %arg2) <{base_dilations = array<i64: 3, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 2, 1>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID2]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // HALO-NEXT:   %[[MUL2:.*]] = stablehlo.multiply %[[CONV2]], %[[C_STRIDE2]] : tensor<i32>
  // HALO-NEXT:   %[[ADD2:.*]] = stablehlo.add %[[MUL2]], %[[C_OFFSET2]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX2:.*]] = stablehlo.maximum %[[ADD2]], %[[C_ZERO3]] : tensor<i32>
  // HALO-NEXT:   %[[C_BD2:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[OFF_OUT:.*]] = stablehlo.remainder %[[MAX2]], %[[C_BD2]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[RW]], %[[OFF_OUT]], %[[C_ZERO2]], sizes = [8, 4]
  // HALO-NEXT:   sdy.return %[[SLICE2]] : tensor<8x4xi32>
  // HALO-NEXT: } : (tensor<4x8xi32>, tensor<i32>) -> tensor<16x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    window_dilations = array<i64: 2, 1>,
    base_dilations = array<i64: 3, 1>,
    padding = dense<[[4, 6], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<4x8xi32>, tensor<i32>) -> tensor<16x8xi32>

  // CHECK: return %[[RES]] : tensor<16x8xi32>
  return %0 : tensor<16x8xi32>
}

// 1D window (4x1) with stride 1 and padding [[5, 4], [0, 0]] on
// tensor<12x8xi32> -> tensor<18x8xi32>, sharded 6-way on dim 0 along @mesh_a6
// (sIn = 2, sOut = 3):
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: Dim 0 has sFootprint = (3 - 1) * 1 + 4 = 6. Shard t needs global
//   elements [3 * t - 5 .. 3 * t], reaching up to 2 shards to the left
//   (leftHops = 2) and 1 shard to the right (rightHops = 1). Since
//   leftHops + rightHops + 1 = 4 < 6, performs 2 left and 1 right
//   collective-permutes, slices a 6x8 footprint at offset max(id * 1 + 5, 0),
//   and runs local reduce_window producing tensor<3x8xi32>.
// CHECK-LABEL: func @reduce_window_1d_multi_hop_left_2_right_1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<12x8xi32>
func.func @reduce_window_1d_multi_hop_left_2_right_1(%arg0: tensor<12x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a6, [{"a"}, {}]>}) -> (tensor<18x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a6, [{"a"}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a6, [{}, {}]> : tensor<12x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[5, 4], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 4, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh_a6, [{"a"}, {}]> : tensor<18x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a6, [{"a"}, {}]>, <@mesh_a6, []>] out_shardings=[<@mesh_a6, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[CONV:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5]]> : tensor<4x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]> : tensor<5x2xi64>}>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL1]], %[[SEL2]], %arg1, %[[SEL3]], dim = 0
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [6, 0], high = [6, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[CONV]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[C_ZERO]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[MAX]], %[[C_ZERO]], sizes = [6, 8]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 4, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD_RW:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD_RW]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<3x8xi32>
  // HALO-NEXT: } : (tensor<12x8xi32>, tensor<i32>) -> tensor<18x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 4, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[5, 4], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a6, [{"a"}, {}]>]>
  } : (tensor<12x8xi32>, tensor<i32>) -> tensor<18x8xi32>

  // CHECK: return %[[RES]] : tensor<18x8xi32>
  return %0 : tensor<18x8xi32>
}

// 2D window (3x3) with stride [1, 1] and padding [[1, 1], [1, 1]] on
// tensor<8x8xi32> -> tensor<8x8xi32>, sharded 2x2 on [{"a"}, {"b"}] (sIn = 4,
// sOut = 4, sFootprint = 6 on both dims):
// - Replication mode: Replicates both dim 0 and dim 1 before reduce_window.
// - HALO mode: Performs sequential 2D HALO exchange inside sdy.manual_computation:
//   first exchanges 1 left and 1 right neighbor along dim 0 (axis "a") and
//   slices from 4x4 to 6x4, then exchanges 1 left and 1 right neighbor along
//   dim 1 (axis "b") and slices to 6x6, and runs local reduce_window with zero
//   local padding producing tensor<4x4xi32>.
// CHECK-LABEL: func @reduce_window_2d_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_2d_sharded(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{padding = dense<1> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L1:.*]] = stablehlo.slice %arg1 [3:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R1:.*]] = stablehlo.slice %arg1 [0:1, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[C_OFFSET1]], %[[C_ZERO1]], sizes = [6, 4]
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM2:.*]] = stablehlo.remainder %[[PID2]], %[[C_MOD2]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[REM2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L2:.*]] = stablehlo.slice %[[SLICE1]] [0:6, 3:4]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R2:.*]] = stablehlo.slice %[[SLICE1]] [0:6, 0:1]
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL4:.*]] = stablehlo.select {{.*}}, %[[CP4]], {{.*}}
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %[[SLICE1]], %[[SEL4]], dim = 1
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %arg2, low = [0, 1], high = [0, 1], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[C_ZERO2]], %[[C_OFFSET2]], sizes = [6, 6]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE2]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<4x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>

  // CHECK: return %[[RES]] : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// 2D window (3x3) on tensor<7x8xi32> -> tensor<7x8xi32> where sharded dim 0
// has size 7, which is not evenly divisible by shardCount = 2:
// - Replication mode: Replicates both dim 0 and dim 1 before reduce_window.
// - HALO mode: High-side pads the input with the init value (%c) to tensor<8x8xi32>
//   (sIn = 4), performs 2D HALO exchange and local reduce_window inside
//   sdy.manual_computation to produce tensor<8x8xi32> (sOut = 4), and trims
//   the global output back to tensor<7x8xi32> via stablehlo.slice [0:7, 0:8].
// CHECK-LABEL: func @reduce_window_1d_indivisible_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<7x8xi32>
func.func @reduce_window_1d_indivisible_dim(%arg0: tensor<7x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<7x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<7x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{padding = dense<1> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<7x8xi32>

  // HALO:      %[[PAD_IN:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT: %[[MANUAL:.*]] = sdy.manual_computation(%[[PAD_IN]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L1:.*]] = stablehlo.slice %arg1 [3:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R1:.*]] = stablehlo.slice %arg1 [0:1, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[C_OFFSET1]], %[[C_ZERO1]], sizes = [6, 4]
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM2:.*]] = stablehlo.remainder %[[PID2]], %[[C_MOD2]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[REM2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L2:.*]] = stablehlo.slice %[[SLICE1]] [0:6, 3:4]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R2:.*]] = stablehlo.slice %[[SLICE1]] [0:6, 0:1]
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL4:.*]] = stablehlo.select {{.*}}, %[[CP4]], {{.*}}
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %[[SLICE1]], %[[SEL4]], dim = 1
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %arg2, low = [0, 1], high = [0, 1], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[C_ZERO2]], %[[C_OFFSET2]], sizes = [6, 6]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE2]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<4x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MANUAL]] [0:7, 0:8]
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<7x8xi32>, tensor<i32>) -> tensor<7x8xi32>

  // CHECK: return %[[RES]] : {{.*}}
  return %0 : tensor<7x8xi32>
}

// Variadic 1D window (3x1) with stride 1 and padding [[1, 1], [0, 0]] on two
// inputs (tensor<8x8xf32>, tensor<8x8xi32>) sharded 2-way on dim 0:
// - Replication mode: Replicates dim 0 of both inputs before reduce_window.
// - HALO mode: Enters a single sdy.manual_computation taking both inputs and both
//   init values (0.0 : f32 and 10 : i32), performs the 1D HALO exchange
//   (leftHops = 1, rightHops = 1, sFootprint = 6) on each input padded with
//   its respective init value, and executes a single variadic local
//   reduce_window returning (tensor<4x8xf32>, tensor<4x8xi32>).
// CHECK-LABEL: func @reduce_window_variadic_1d_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {{.*}}, %[[ARG1:.*]]: tensor<8x8xi32>
func.func @reduce_window_variadic_1d_sharded(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<8x8xf32>, tensor<8x8xi32>) {
  // CHECK:      %[[C0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<10> : tensor<i32>
  %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %c1 = stablehlo.constant dense<10> : tensor<i32>

  // REPL:      %[[RESHARD_IN0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[RESHARD_IN1:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]]:2 = "stablehlo.reduce_window"(%[[RESHARD_IN0]], %[[RESHARD_IN1]], %[[C0]], %[[C1]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<i32>):
  // REPL-NEXT:   %[[ADD0:.*]] = stablehlo.add %arg2, %arg4 : tensor<f32>
  // REPL-NEXT:   %[[ADD1:.*]] = stablehlo.add %arg3, %arg5 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD0]], %[[ADD1]] : tensor<f32>, tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES0:.*]] = sdy.reshard %[[RW]]#0 <@mesh, [{"a"}, {}]> : tensor<8x8xf32>
  // REPL-NEXT: %[[RES1:.*]] = sdy.reshard %[[RW]]#1 <@mesh, [{"a"}, {}]> : tensor<8x8xi32>

  // HALO:      %[[MANUAL:.*]]:2 = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[C0]], %[[C1]])
  // HALO-SAME:     in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"a"}, {}]>, <@mesh, []>, <@mesh, []>]
  // HALO-SAME:     out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{"a"}, {}]>]
  // HALO-SAME:     manual_axes={"a"}
  // HALO-SAME:     (%[[ARG0_LOC:.*]]: tensor<4x8xf32>, %[[ARG1_LOC:.*]]: tensor<4x8xi32>, %[[C0_LOC:.*]]: tensor<f32>, %[[C1_LOC:.*]]: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L1:.*]] = stablehlo.slice %[[ARG0_LOC]] [3:4, 0:8]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R1:.*]] = stablehlo.slice %[[ARG0_LOC]] [0:1, 0:8]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %[[ARG0_LOC]], %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %[[C0_LOC]], low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[C_OFFSET1]], %[[C_ZERO1]], sizes = [6, 8]
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV2:.*]] = stablehlo.divide %[[PID2]], %[[C_DIV2]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[DIV2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L2:.*]] = stablehlo.slice %[[ARG1_LOC]] [3:4, 0:8]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R2:.*]] = stablehlo.slice %[[ARG1_LOC]] [0:1, 0:8]
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL4:.*]] = stablehlo.select {{.*}}, %[[CP4]], {{.*}}
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %[[ARG1_LOC]], %[[SEL4]], dim = 0
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %[[C1_LOC]], low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[C_OFFSET2]], %[[C_ZERO2]], sizes = [6, 8]
  // HALO-NEXT:   %[[RW_LOC:.*]]:2 = "stablehlo.reduce_window"(%[[SLICE1]], %[[SLICE2]], %[[C0_LOC]], %[[C1_LOC]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<f32>, %arg7: tensor<i32>, %arg8: tensor<f32>, %arg9: tensor<i32>):
  // HALO-NEXT:     %[[ADD0:.*]] = stablehlo.add %arg6, %arg8 : tensor<f32>
  // HALO-NEXT:     %[[ADD1:.*]] = stablehlo.add %arg7, %arg9 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD0]], %[[ADD1]] : tensor<f32>, tensor<i32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>]>} : (tensor<6x8xf32>, tensor<6x8xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x8xf32>, tensor<4x8xi32>)
  // HALO-NEXT:   sdy.return %[[RW_LOC]]#0, %[[RW_LOC]]#1 : tensor<4x8xf32>, tensor<4x8xi32>
  // HALO-NEXT: } : (tensor<8x8xf32>, tensor<8x8xi32>, tensor<f32>, tensor<i32>) -> (tensor<8x8xf32>, tensor<8x8xi32>)
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %c0, %c1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<i32>):
      %1 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %2 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {}]>, #sdy.sharding<@mesh, [{"a"}, {}]>]>
  } : (tensor<8x8xf32>, tensor<8x8xi32>, tensor<f32>, tensor<i32>) -> (tensor<8x8xf32>, tensor<8x8xi32>)

  // REPL: return %[[RES0]], %[[RES1]] : tensor<8x8xf32>, tensor<8x8xi32>
  // HALO: return %[[MANUAL]]#0, %[[MANUAL]]#1 : tensor<8x8xf32>, tensor<8x8xi32>
  return %0#0, %0#1 : tensor<8x8xf32>, tensor<8x8xi32>
}

// Passthrough 1x1 window with stride 1 and zero padding on tensor<8x8xi32>,
// sharded 2x2 on [{"a"}, {"b"}]:
// - Both modes: All dimensions are passthrough (kPassThrough factors
//   in the sharding rule), so no communication, resharding, or manual
//   computation is needed; the op is left unchanged.
// CHECK-LABEL: func @reduce_window_1d_passthrough_only
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_1d_passthrough_only(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK:      %[[RW:.*]] = "stablehlo.reduce_window"(%[[ARG0]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 1, 1>, window_strides = array<i64: 1, 1>}> ({
  // CHECK-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // CHECK-NEXT: })
  // CHECK-NEXT: return %[[RW]] : tensor<8x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>

  return %0 : tensor<8x8xi32>
}

// Mixed sharding with a 3x1 window (stride 1, zero padding) on tensor<8x8xi32>
// -> tensor<6x8xi32>, sharded 2x2 on [{"a"}, {"b"}]:
// - Replication mode: Dim 0 is a permutation factor while dim 1 (window size 1) is
//   passthrough, so only dim 0 is replicated ([{}, {"b"}]).
// - HALO mode: Performs HALO exchange only on dim 0 (sFootprint = 5, leftHops = 1,
//   rightHops = 1) while leaving sharded dim 1 untouched (local size 4),
//   slicing tensor<5x4xi32> and producing local output tensor<3x4xi32>.
// CHECK-LABEL: func @reduce_window_2d_mixed_sharding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_2d_mixed_sharding(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [3:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg1 [0:1, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[CONV1]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[C_ZERO1]] : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[MAX1]], %[[C_ZERO1]], sizes = [5, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE1]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<3x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x8xi32>

  // CHECK: return %[[RES]] : tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// Large 1D window (size 7, stride 1, padding [[3, 3]]) on tensor<8xi32>,
// sharded 4-way on @mesh_a_4 (sIn = 2, sOut = 2):
// - Both modes: sFootprint = (2 - 1) * 1 + 7 = 8 requires leftHops = 2
//   and rightHops = 2. Because leftHops + rightHops + 1 = 5 >= shardCount = 4
//   and max(leftHops, rightHops) > 1, canUseHalo is false and HALO falls back
//   to replicating dim 0.
// CHECK-LABEL: func @reduce_window_large_window_replication_fallback
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi32>
func.func @reduce_window_large_window_replication_fallback(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<8xi32>
  // CHECK-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{padding = dense<3> : tensor<1x2xi64>, window_dimensions = array<i64: 7>, window_strides = array<i64: 1>}> ({
  // CHECK-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // CHECK-NEXT: })
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh_a_4, [{"a"}]> : tensor<8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 7>,
    window_strides = array<i64: 1>,
    padding = dense<[[3, 3]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}]>]>
  } : (tensor<8xi32>, tensor<i32>) -> tensor<8xi32>

  // CHECK: return %[[RES]] : tensor<8xi32>
  return %0 : tensor<8xi32>
}

// 1D window (size 3, stride 2, base_dilations = [3], zero padding) on
// tensor<4xi32> -> tensor<4xi32>, sharded 2-way on @mesh (sIn = 2, sOut = 2):
// - Both modes: sOut * stride = 4 is not divisible by
//   baseDilation = 3, and stride = 2 != 1, so the dilation phase shifts across
//   shards and cannot be corrected by a contiguous post-op slice. HALO marks
//   canUseHalo = false and falls back to replicating dim 0.
// CHECK-LABEL: func @reduce_window_base_dilation_unaligned_stride_replication_fallback
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xi32>
func.func @reduce_window_base_dilation_unaligned_stride_replication_fallback(%arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<4xi32>
  // CHECK-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]]) <{base_dilations = array<i64: 3>, padding = dense<0> : tensor<1x2xi64>, window_dimensions = array<i64: 3>, window_strides = array<i64: 2>}> ({
  // CHECK-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // CHECK-NEXT: })
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}]> : tensor<4xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3>,
    window_strides = array<i64: 2>,
    base_dilations = array<i64: 3>,
    padding = dense<[[0, 0]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>
  } : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>

  // CHECK: return %[[RES]] : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// 1D window (size 6, stride 2, base_dilations = [3], padding [[5, 2]]) on
// tensor<8xi32> -> tensor<12xi32>, sharded 4-way on @mesh_a_4 (sIn = 2,
// sOut = 3):
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: Even though stride = 2 > 1 and baseDilation = 3 > 1,
//   sOut * stride = 6 is divisible by baseDilation = 3, so all shards share
//   the same dilation phase (localPadLow = 5 mod 3 = 2, effectivePadLow = 3,
//   sFootprint = 3, localPadHigh = 1). Performs 1 left-hop HALO exchange,
//   slices tensor<3xi32> at undilated offset dilatedOffset / 3, and runs local
//   reduce_window with residual local padding [[2, 1]] to directly produce
//   tensor<3xi32> without a post-op slice.
// CHECK-LABEL: func @reduce_window_base_dilation_phase_aligned_stride
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi32>
func.func @reduce_window_base_dilation_phase_aligned_stride(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) -> (tensor<12xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{base_dilations = array<i64: 3>, padding = dense<[[5, 2]]> : tensor<1x2xi64>, window_dimensions = array<i64: 6>, window_strides = array<i64: 2>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh_a_4, [{"a"}]> : tensor<12xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a_4, [{"a"}]>, <@mesh_a_4, []>] out_shardings=[<@mesh_a_4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<2xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[CONV:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [1:2]
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>}>
  // HALO:        %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [1], high = [0], interior = [0]
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_UNDIL:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[C_UNDIL]], sizes = [3]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE]], %arg2)
  // HALO-SAME{LITERAL}: <{base_dilations = array<i64: 3>, padding = dense<[[2, 1]]> : tensor<1x2xi64>, window_dimensions = array<i64: 6>, window_strides = array<i64: 2>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD_RW:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD_RW]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<3xi32>
  // HALO-NEXT: } : (tensor<8xi32>, tensor<i32>) -> tensor<12xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 6>,
    window_strides = array<i64: 2>,
    base_dilations = array<i64: 3>,
    padding = dense<[[5, 2]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}]>]>
  } : (tensor<8xi32>, tensor<i32>) -> tensor<12xi32>

  // CHECK: return %[[RES]] : tensor<12xi32>
  return %0 : tensor<12xi32>
}

// Variadic 2D window (3x3, stride 1, padding [[1, 1], [1, 1]]) on two inputs
// (tensor<8x8xi32>, tensor<8x8xi32>) sharded 2x2 on [{"a"}, {"b"}]:
// - Replication mode: Replicates both dims of both inputs before reduce_window.
// - HALO mode: Performs sequential 2D HALO exchange (4x4 -> 6x4 -> 6x6) on each of
//   the two inputs using its respective init value (0 and 10), then runs a
//   single variadic local reduce_window producing
//   (tensor<4x4xi32>, tensor<4x4xi32>).
// CHECK-LABEL: func @reduce_window_variadic_2d_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}, %[[ARG1:.*]]: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
func.func @reduce_window_variadic_2d_sharded(
    %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
    -> (tensor<8x8xi32>, tensor<8x8xi32>) {
  // CHECK:      %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<10> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<10> : tensor<i32>

  // REPL:      %[[RESHARD_IN0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RESHARD_IN1:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]]:2 = "stablehlo.reduce_window"(%[[RESHARD_IN0]], %[[RESHARD_IN1]], %[[C0]], %[[C1]]) <{padding = dense<1> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
  // REPL-NEXT:   %[[ADD0:.*]] = stablehlo.add %arg2, %arg4 : tensor<i32>
  // REPL-NEXT:   %[[ADD1:.*]] = stablehlo.add %arg3, %arg5 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD0]], %[[ADD1]] : tensor<i32>, tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES0:.*]] = sdy.reshard %[[RW]]#0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RES1:.*]] = sdy.reshard %[[RW]]#1 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>

  // HALO:      %[[MANUAL:.*]]:2 = sdy.manual_computation(%[[ARG0]], %[[ARG1]], %[[C0]], %[[C1]])
  // HALO-SAME:     in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>, <@mesh, []>, <@mesh, []>]
  // HALO-SAME:     out_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {"b"}]>]
  // HALO-SAME:     manual_axes={"a", "b"}
  // HALO-SAME:     (%[[ARG0_LOC:.*]]: tensor<4x4xi32>, %[[ARG1_LOC:.*]]: tensor<4x4xi32>, %[[C0_LOC:.*]]: tensor<i32>, %[[C1_LOC:.*]]: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L1:.*]] = stablehlo.slice %[[ARG0_LOC]] [3:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R1:.*]] = stablehlo.slice %[[ARG0_LOC]] [0:1, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %[[ARG0_LOC]], %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %[[C0_LOC]], low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[C_OFFSET1]], %[[C_ZERO1]], sizes = [6, 4]
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM2:.*]] = stablehlo.remainder %[[PID2]], %[[C_MOD2]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[REM2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L2:.*]] = stablehlo.slice %[[SLICE1]] [0:6, 3:4]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R2:.*]] = stablehlo.slice %[[SLICE1]] [0:6, 0:1]
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL4:.*]] = stablehlo.select {{.*}}, %[[CP4]], {{.*}}
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %[[SLICE1]], %[[SEL4]], dim = 1
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %[[C0_LOC]], low = [0, 1], high = [0, 1], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[C_ZERO2]], %[[C_OFFSET2]], sizes = [6, 6]
  // HALO-NEXT:   %[[PID3:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV3:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV3:.*]] = stablehlo.divide %[[PID3]], %[[C_DIV3]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV3:.*]] = stablehlo.convert %[[DIV3]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L3:.*]] = stablehlo.slice %[[ARG1_LOC]] [3:4, 0:4]
  // HALO-NEXT:   %[[CP5:.*]] = "stablehlo.collective_permute"(%[[SLICE_L3]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL5:.*]] = stablehlo.select {{.*}}, %[[CP5]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R3:.*]] = stablehlo.slice %[[ARG1_LOC]] [0:1, 0:4]
  // HALO-NEXT:   %[[CP6:.*]] = "stablehlo.collective_permute"(%[[SLICE_R3]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL6:.*]] = stablehlo.select {{.*}}, %[[CP6]], {{.*}}
  // HALO-NEXT:   %[[CONCAT3:.*]] = stablehlo.concatenate %[[SEL5]], %[[ARG1_LOC]], %[[SEL6]], dim = 0
  // HALO-NEXT:   %[[PAD3:.*]] = stablehlo.pad %[[CONCAT3]], %[[C1_LOC]], low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET3:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO3:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE3:.*]] = stablehlo.dynamic_slice %[[PAD3]], %[[C_OFFSET3]], %[[C_ZERO3]], sizes = [6, 4]
  // HALO-NEXT:   %[[PID4:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD4:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM4:.*]] = stablehlo.remainder %[[PID4]], %[[C_MOD4]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV4:.*]] = stablehlo.convert %[[REM4]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L4:.*]] = stablehlo.slice %[[SLICE3]] [0:6, 3:4]
  // HALO-NEXT:   %[[CP7:.*]] = "stablehlo.collective_permute"(%[[SLICE_L4]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL7:.*]] = stablehlo.select {{.*}}, %[[CP7]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R4:.*]] = stablehlo.slice %[[SLICE3]] [0:6, 0:1]
  // HALO-NEXT:   %[[CP8:.*]] = "stablehlo.collective_permute"(%[[SLICE_R4]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL8:.*]] = stablehlo.select {{.*}}, %[[CP8]], {{.*}}
  // HALO-NEXT:   %[[CONCAT4:.*]] = stablehlo.concatenate %[[SEL7]], %[[SLICE3]], %[[SEL8]], dim = 1
  // HALO-NEXT:   %[[PAD4:.*]] = stablehlo.pad %[[CONCAT4]], %[[C1_LOC]], low = [0, 1], high = [0, 1], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO4:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE4:.*]] = stablehlo.dynamic_slice %[[PAD4]], %[[C_ZERO4]], %[[C_OFFSET4]], sizes = [6, 6]
  // HALO-NEXT:   %[[RW_LOC:.*]]:2 = "stablehlo.reduce_window"(%[[SLICE2]], %[[SLICE4]], %[[C0_LOC]], %[[C1_LOC]]) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg6: tensor<i32>, %arg7: tensor<i32>, %arg8: tensor<i32>, %arg9: tensor<i32>):
  // HALO-NEXT:     %[[ADD0:.*]] = stablehlo.add %arg6, %arg8 : tensor<i32>
  // HALO-NEXT:     %[[ADD1:.*]] = stablehlo.add %arg7, %arg9 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD0]], %[[ADD1]] : tensor<i32>, tensor<i32>
  // HALO-NEXT:   }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>]>} : (tensor<6x6xi32>, tensor<6x6xi32>, tensor<i32>, tensor<i32>) -> (tensor<4x4xi32>, tensor<4x4xi32>)
  // HALO-NEXT:   sdy.return %[[RW_LOC]]#0, %[[RW_LOC]]#1 : tensor<4x4xi32>, tensor<4x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<i32>, tensor<i32>) -> (tensor<8x8xi32>, tensor<8x8xi32>)
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %c0, %c1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
      %1 = stablehlo.add %arg2, %arg4 : tensor<i32>
      %2 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %1, %2 : tensor<i32>, tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 3>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>, #sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<i32>, tensor<i32>) -> (tensor<8x8xi32>, tensor<8x8xi32>)

  // REPL: return %[[RES0]], %[[RES1]] : tensor<8x8xi32>, tensor<8x8xi32>
  // HALO: return %[[MANUAL]]#0, %[[MANUAL]]#1 : tensor<8x8xi32>, tensor<8x8xi32>
  return %0#0, %0#1 : tensor<8x8xi32>, tensor<8x8xi32>
}

// 1D window (3x1, stride [2, 1], padding [[1, 1], [0, 0]]) on
// tensor<7x8xi32> -> tensor<4x8xi32>, sharded 2x2 on [{"a"}, {"b"}]:
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: Input dim 0 has size 7 (not divisible by 2) while output dim 0 has
//   size 4 (already divisible by 2, sOut = 2). High-side pads only the input
//   from 7x8 to 8x8 (sIn = 4), performs 1 left-hop HALO exchange
//   (sFootprint = 5) and local reduce_window producing tensor<2x4xi32> (global
//   4x8), with no output trimming needed.
// CHECK-LABEL: func @reduce_window_1d_indivisible_dim_padding_and_stride
// CHECK-SAME: (%[[ARG0:.*]]: tensor<7x8xi32>
func.func @reduce_window_1d_indivisible_dim_padding_and_stride(%arg0: tensor<7x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<7x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 2, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<4x8xi32>

  // HALO:      %[[PAD_IN:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[PAD_IN]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [3:4, 0:4]
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [1, 0], high = [0, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[C_OFFSET]], %[[C_ZERO]], sizes = [5, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 2, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD_RW:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD_RW]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<2x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<4x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<7x8xi32>, tensor<i32>) -> tensor<4x8xi32>

  // CHECK: return %[[RES]] : tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// Asymmetric causal padding [[2, 0], [0, 0]] with a 3x1 window (stride 1) on
// tensor<8x8xi32> -> tensor<8x8xi32>, sharded 2x2 on [{"a"}, {"b"}] (sIn = 4,
// sOut = 4):
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: Shard 0 needs global elements [-2..3] (within shard 0 plus low
//   padding) and shard 1 needs [2..7] (elements [2, 3] from shard 0 and [4..7]
//   from shard 1). Only a left neighbor exchange is needed (leftHops = 1,
//   rightHops = 0, sFootprint = 6), slicing tensor<6x4xi32> at offset
//   max(id * 0 + 2, 0) (baseOffset = -2 + (2 + 2) = 2).
// CHECK-LABEL: func @reduce_window_1d_asymmetric_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_1d_asymmetric_padding(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[2, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [2:4, 0:4]
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [2, 0], high = [0, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[C_OFFSET]], %[[C_ZERO]], sizes = [6, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD_RW:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD_RW]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<4x4xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[2, 0], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>

  // CHECK: return %[[RES]] : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// Compound axis sharding [{"a", "b"}, {}] on @mesh (a=2, b=2, so dim 0 is
// 4-way sharded with sIn = 2, sOut = 2) with a 3x1 window, stride 1, and
// padding [[1, 1], [0, 0]]:
// - Replication mode: Replicates dim 0 before reduce_window.
// - HALO mode: Computes the linearized 4-way partition index along {"a", "b"}
//   (id = ((pid / 2) % 2) * 2 + (pid % 2)), exchanges 1 left and 1 right
//   neighbor across the 4-device chain, slices tensor<4x8xi32>
//   (sFootprint = 4), and runs local reduce_window producing tensor<2x8xi32>.
// CHECK-LABEL: func @reduce_window_1d_compound_axis_sharding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_1d_compound_axis_sharding(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}) -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a", "b"}, {}]> : tensor<8x8xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a", "b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a", "b"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<2x8xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM2:.*]] = stablehlo.remainder %[[PID]], %[[C_MOD2]] : tensor<ui32>
  // HALO-NEXT:   %[[C_MUL:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[MUL_ID:.*]] = stablehlo.multiply %[[DIV]], %[[C_MUL]] : tensor<ui32>
  // HALO-NEXT:   %[[ID_U:.*]] = stablehlo.add %[[MUL_ID]], %[[REM2]] : tensor<ui32>
  // HALO-NEXT:   %[[ID:.*]] = stablehlo.convert %[[ID_U]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [1:2, 0:8]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R:.*]] = stablehlo.slice %arg1 [0:1, 0:8]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [2, 1], [3, 2]]> : tensor<3x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[C_OFFSET]], %[[C_ZERO]], sizes = [4, 8]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE]], %arg2) <{padding = dense<0> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD_RW:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD_RW]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<2x8xi32>
  // HALO-NEXT: } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a", "b"}, {}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>

  // CHECK: return %[[RES]] : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// 4D NHWC tensor<2x8x8x4xi32> sharded on spatial dimensions [{}, {"a"}, {"b"},
// {}] with window [1, 3, 3, 1], stride 1, and padding
// [[0, 0], [1, 1], [1, 1], [0, 0]]:
// - Replication mode: Replicates spatial dims 1 and 2 before reduce_window.
// - HALO mode: Performs 2D HALO exchange across the middle spatial dimensions
//   (dim 1 on axis "a" and dim 2 on axis "b", expanding local shape
//   2x4x4x4 -> 2x6x4x4 -> 2x6x6x4) while keeping batch (dim 0) and feature
//   (dim 3) dimensions unsharded, then executes local reduce_window producing
//   tensor<2x4x4x4xi32>.
// CHECK-LABEL: func @reduce_window_4d_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8x8x4xi32>
func.func @reduce_window_4d_sharded(%arg0: tensor<2x8x8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {"b"}, {}]>}) -> (tensor<2x8x8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {"b"}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<2x8x8x4xi32>
  // REPL-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL-SAME{LITERAL}: <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
  // REPL-NEXT: ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
  // REPL-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<i32>
  // REPL-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // REPL-NEXT: })
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{}, {"a"}, {"b"}, {}]> : tensor<2x8x8x4xi32>

  // HALO:      %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{}, {"a"}, {"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {"a"}, {"b"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<2x4x4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L1:.*]] = stablehlo.slice %arg1 [0:2, 3:4, 0:4, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R1:.*]] = stablehlo.slice %arg1 [0:2, 0:1, 0:4, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE_R1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 1
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [0, 1, 0, 0], high = [0, 1, 0, 0], interior = [0, 0, 0, 0]
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[C_ZERO1]], %[[C_OFFSET1]], %[[C_ZERO1]], %[[C_ZERO1]], sizes = [2, 6, 4, 4]
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM2:.*]] = stablehlo.remainder %[[PID2]], %[[C_MOD2]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[REM2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L2:.*]] = stablehlo.slice %[[SLICE1]] [0:2, 0:6, 3:4, 0:4]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_L2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[SLICE_R2:.*]] = stablehlo.slice %[[SLICE1]] [0:2, 0:6, 0:1, 0:4]
  // HALO-NEXT:   %[[CP4:.*]] = "stablehlo.collective_permute"(%[[SLICE_R2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}>
  // HALO:        %[[SEL4:.*]] = stablehlo.select {{.*}}, %[[CP4]], {{.*}}
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %[[SLICE1]], %[[SEL4]], dim = 2
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %arg2, low = [0, 0, 1, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0]
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[C_ZERO2]], %[[C_ZERO2]], %[[C_OFFSET2]], %[[C_ZERO2]], sizes = [2, 6, 6, 4]
  // HALO-NEXT:   %[[RW:.*]] = "stablehlo.reduce_window"(%[[SLICE2]], %arg2) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
  // HALO-NEXT:   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
  // HALO-NEXT:     %[[ADD:.*]] = stablehlo.add %arg3, %arg4 : tensor<i32>
  // HALO-NEXT:     stablehlo.return %[[ADD]] : tensor<i32>
  // HALO-NEXT:   })
  // HALO-NEXT:   sdy.return %[[RW]] : tensor<2x4x4x4xi32>
  // HALO-NEXT: } : (tensor<2x8x8x4xi32>, tensor<i32>) -> tensor<2x8x8x4xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 1, 1, 1>,
    padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {"a"}, {"b"}, {}]>]>
  } : (tensor<2x8x8x4xi32>, tensor<i32>) -> tensor<2x8x8x4xi32>

  // CHECK: return %[[RES]] : tensor<2x8x8x4xi32>
  return %0 : tensor<2x8x8x4xi32>
}

// Single-use operand reshard from [{}, {"b"}] to [{"a"}, {"b"}] and single-use
// result reshard back to [{}, {"b"}]:
// - Both modes: Falls back to replication so CanonicalizerPass(kReshardLabel)
//   folds both reshards to zero collectives.
// CHECK-LABEL: func @reduce_window_back_to_back_reshard_replication_fallback
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @reduce_window_back_to_back_reshard_replication_fallback(
    %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
    -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>
  // CHECK:      %[[IN_RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>
  // CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[IN_RESHARD]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // CHECK-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // CHECK:      %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>
  // CHECK-NEXT: %[[OUT_RESHARD:.*]] = sdy.reshard %[[RES]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // CHECK-NEXT: return %[[OUT_RESHARD]] : tensor<8x8xi32>
  %in_reshard = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xi32>
  %0 = "stablehlo.reduce_window"(%in_reshard, %c) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<[[1, 1], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  %out_reshard = sdy.reshard %0 <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  return %out_reshard : tensor<8x8xi32>
}

// 2-sided halo exchange with sFootprint = 5 > inputDimSize = 4 where the
// single user of reduce_window immediately reshards to replicate the window
// dimension:
// - Both modes: Falls back to replication so the input is replicated via
//   all-gather instead of performing 2-sided halo exchange plus output
//   all-gather.
// CHECK-LABEL: func @reduce_window_small_two_side_halo_single_use_reshard_replication_fallback
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x2xf32>
func.func @reduce_window_small_two_side_halo_single_use_reshard_replication_fallback(
    %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %c = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:      %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<4x2xf32>
  // CHECK-NEXT: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // CHECK:      %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {}]> : tensor<2x2xf32>
  // CHECK-NEXT: %[[OUT_RESHARD:.*]] = sdy.reshard %[[RES]] <@mesh, [{}, {}]> : tensor<2x2xf32>
  // CHECK-NEXT: return %[[OUT_RESHARD]] : tensor<2x2xf32>
  %0 = "stablehlo.reduce_window"(%arg0, %c) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
  }) {
    window_dimensions = array<i64: 5, 1>,
    window_strides = array<i64: 3, 1>,
    padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {}]>]>
  } : (tensor<4x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  %out_reshard = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<2x2xf32>
  return %out_reshard : tensor<2x2xf32>
}

