// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// CHECK-LABEL: func @swap_two_dim_shardings
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(2)2}]>})
// CHECK-SAME: -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(2)2}, {"x"}]>})
func.func @swap_two_dim_shardings(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y":(2)2}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y":(2)2}, {"x"}]>}) {
  // CHECK: %[[RES:.*]] = "stablehlo.collective_permute"(%[[ARG0]]) <{
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // CHECK-SAME{LITERAL}: source_target_pairs = dense<[[0, 0], [1, 1], [2, 4], [3, 5], [4, 2], [5, 3], [6, 6], [7, 7]]> : tensor<8x2xi64>
  // CHECK-SAME: }> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2_4, [{"y":(2)2}, {"x"}]> : tensor<4x8xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reverse_two_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x", "y"}]>})
// CHECK-SAME: -> (tensor<4x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y", "x"}]>})
func.func @reverse_two_axes(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x", "y"}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y", "x"}]>}) {
  // CHECK:      %[[RES:.*]] = "stablehlo.collective_permute"(%[[ARG0]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
  // CHECK-SAME{LITERAL}:   source_target_pairs = dense<[[0, 0], [1, 4], [2, 1], [3, 5], [4, 2], [5, 6], [6, 3], [7, 7]]> : tensor<8x2xi64>
  // CHECK-SAME: }> : (tensor<4x1xf32>) -> tensor<4x1xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2_4, [{}, {"y", "x"}]> : tensor<4x8xf32>
  // CHECK:      return %[[RES]] : tensor<4x1xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @regroup_sharding_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>})
// CHECK-SAME: -> (tensor<2x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y":(2)2, "x"}, {"z", "y":(1)2}]>})
func.func @regroup_sharding_axes(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>})
    -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y":(2)2, "x"}, {"z", "y":(1)2}]>}) {
  // CHECK:      %[[RES:.*]] = "stablehlo.collective_permute"(%[[ARG0]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
  // CHECK-SAME{LITERAL}:   source_target_pairs = dense<[[0, 0], [1, 8], [2, 1], [3, 9], [4, 2], [5, 10], [6, 3], [7, 11], [8, 4], [9, 12], [10, 5], [11, 13], [12, 6], [13, 14], [14, 7], [15, 15]]> : tensor<16x2xi64>
  // CHECK-SAME: }> : (tensor<2x1xf32>) -> tensor<2x1xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2_4_2, [{"y":(2)2, "x"}, {"z", "y":(1)2}]> : tensor<8x4xf32>
  // CHECK:      return %[[RES]] : tensor<2x1xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @replica_axis_changed
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y"}, {"x"}]>})
// CHECK-SAME: -> (tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y":(2)2, "z"}, {"y":(1)2}]>}) {
func.func @replica_axis_changed(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y"}, {"x"}]>})
    -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y":(2)2, "z"}, {"y":(1)2}]>}) {
  // CHECK:      %[[RES:.*]] = "stablehlo.collective_permute"(%[[ARG0]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>,
  // CHECK-SAME{LITERAL}:   source_target_pairs = dense<[[0, 0], [1, 8], [2, 4], [3, 12], [4, 1], [5, 9], [6, 5], [7, 13], [8, 2], [9, 10], [10, 6], [11, 14], [12, 3], [13, 11], [14, 7], [15, 15]]> : tensor<16x2xi64>
  // CHECK-SAME: }> : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2_4_2, [{"y":(2)2, "z"}, {"y":(1)2}]> : tensor<8x4xf32>
  // CHECK:      return %[[RES]] : tensor<2x2xf32>
  return %0 : tensor<8x4xf32>
}
