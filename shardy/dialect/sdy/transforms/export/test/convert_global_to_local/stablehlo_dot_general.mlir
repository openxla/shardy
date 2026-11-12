// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @not_shard_contracting_dims(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<2x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
// CHECK-SAME: -> (tensor<2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
func.func @not_shard_contracting_dims(
  %arg0: tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>},
  %arg1: tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>})
  -> (tensor<4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
  // CHECK: %[[RES:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}]>]>}
  // CHECK-SAME: : (tensor<2x16x8xf32>, tensor<2x8x16xf32>) -> tensor<2x16x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}]>]>}
  : (tensor<4x16x8xf32>, tensor<4x8x16xf32>) -> tensor<4x16x16xf32>

  // CHECK: return %[[RES]] : tensor<2x16x16xf32>
  return %0 : tensor<4x16x16xf32>
}

// CHECK-LABEL: func @shard_contracting_dims
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x2x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}, {}]>})
// CHECK-SAME: -> (tensor<2x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
func.func @shard_contracting_dims(
  %arg0: tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {"y"}]>},
  %arg1: tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}, {}]>})
  -> (tensor<4x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}], unreduced={"y"}>]>}
  // CHECK-SAME: : (tensor<2x16x2xf32>, tensor<2x2x16xf32>) -> tensor<2x16x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"x"}, {}, {}], unreduced={"y"}>]>}
  : (tensor<4x16x8xf32>, tensor<4x8x16xf32>) -> tensor<4x16x16xf32>

  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[DOT]])
  // CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
  // CHECK-SAME: use_global_device_ids
  // CHECK: ^bb0(%[[ACC:.*]]: tensor<f32>, %[[UPD:.*]]: tensor<f32>):
  // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
  // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK: }) : (tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  %1 = sdy.all_reduce {"y"} %0 out_sharding=<@mesh_2_4, [{"x"}, {}, {}]> : tensor<4x16x16xf32>

  // CHECK: return %[[RES]] : tensor<2x16x16xf32>
  return %1 : tensor<4x16x16xf32>
}

// CHECK-LABEL: func @shard_contracting_dims_with_nonzero_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {"y"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x2x7xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}, {}]>})
// CHECK-SAME: -> (tensor<2x5x7xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
func.func @shard_contracting_dims_with_nonzero_pad(
  %arg0: tensor<4x3x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {"y"}]>},
  %arg1: tensor<4x8x7xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}, {}]>})
  -> (tensor<4x5x7xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<2x3x2xf32>, tensor<f32>) -> tensor<2x5x2xf32>
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %pad = stablehlo.pad %arg0, %cst, low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {"y"}]>]>}
    : (tensor<4x3x8xf32>, tensor<f32>) -> tensor<4x5x8xf32>

  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[PAD]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}], unreduced={"y"}>]>}
  // CHECK-SAME: : (tensor<2x5x2xf32>, tensor<2x2x7xf32>) -> tensor<2x5x7xf32>
  %0 = stablehlo.dot_general %pad, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_4, [{"x"}, {}, {}], unreduced={"y"}>]>}
  : (tensor<4x5x8xf32>, tensor<4x8x7xf32>) -> tensor<4x5x7xf32>

  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[DOT]])
  %1 = sdy.all_reduce {"y"} %0 out_sharding=<@mesh_2_4, [{"x"}, {}, {}]> : tensor<4x5x7xf32>

  // CHECK: return %[[RES]] : tensor<2x5x7xf32>
  return %1 : tensor<4x5x7xf32>
}

