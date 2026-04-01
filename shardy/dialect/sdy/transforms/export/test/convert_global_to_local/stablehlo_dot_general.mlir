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
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}]>]>}
  // CHECK-SAME: : (tensor<2x16x2xf32>, tensor<2x2x16xf32>) -> tensor<2x16x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}, {}]>]>}
  : (tensor<4x16x8xf32>, tensor<4x8x16xf32>) -> tensor<4x16x16xf32>

  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[DOT]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>
  // CHECK-SAME: use_global_device_ids
  // CHECK: ^bb0(%[[ACC:.*]]: tensor<f32>, %[[UPD:.*]]: tensor<f32>):
  // CHECK:   %[[ADD:.*]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
  // CHECK:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK: }) : (tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
  %1 = sdy.all_reduce {"y"} %0 out_sharding=<@mesh_2_4, [{"x"}, {}, {}]> : tensor<4x16x16xf32>

  // CHECK: return %[[RES]] : tensor<2x16x16xf32>
  return %1 : tensor<4x16x16xf32>
}
