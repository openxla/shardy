// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,V1
// RUN: sdy_opt %s -sdy-convert-global-to-local='enable-rgv3=true' | FileCheck %s --check-prefixes=CHECK,V3

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// CHECK-LABEL: func @one_param_move_suffix
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>})
func.func @one_param_move_suffix(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {}]>})
    -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>}) {
  // CHECK: %[[RESULT:.*]] = "stablehlo.all_to_all"(%[[ARG0]]) <{
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
  // CHECK-SAME: concat_dimension = 0 : i64
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"y">]>
  // CHECK-SAME: split_count = 4 : i64
  // CHECK-SAME: split_dimension = 1 : i64
  // CHECK-SAME: }> : (tensor<1x32xf32>) -> tensor<4x8xf32>
  %0 = sdy.all_to_all [{"y"}: 0->1] %arg0 out_sharding=<@mesh_2_4_2, [{"x"}, {"y"}]> : tensor<8x32xf32>
  // CHECK: return %[[RESULT]] : tensor<4x8xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @one_param_move_all_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"y", "z"}]>})
// CHECK-SAME: -> (tensor<1x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y", "z"}, {}]>})
func.func @one_param_move_all_axes(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"y", "z"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y", "z"}, {}]>}) {
  // CHECK: %[[RESULT:.*]] = "stablehlo.all_to_all"(%[[ARG0]]) <{
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>
  // CHECK-SAME: concat_dimension = 1 : i64
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"y">, #sdy<axis_ref"z">]>
  // CHECK-SAME: split_count = 8 : i64
  // CHECK-SAME: split_dimension = 0 : i64
  // CHECK-SAME: }> : (tensor<8x2xf32>) -> tensor<1x16xf32>
  %0 = sdy.all_to_all [{"y", "z"}: 1->0] %arg0 out_sharding=<@mesh_2_4_2, [{"y", "z"}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[RESULT]] : tensor<1x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @two_params_move_suffix
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x1x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y":(2)2}, {"z"}, {}]>})
// CHECK-SAME: -> (tensor<1x4x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"z"}, {"x"}, {}, {"y":(2)2}]>}) {
func.func @two_params_move_suffix(%arg0: tensor<2x8x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y":(2)2}, {"z"}, {}]>})
    -> (tensor<2x8x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"z"}, {"x"}, {}, {"y":(2)2}]>}) {
  // CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<2x2x1x4xf32>) -> tensor<2x1x2x1x2x2xf32>
  // CHECK: %[[TRANSPOSE0:.*]] = stablehlo.transpose %[[RESHAPE0]], dims = [0, 4, 1, 2, 3, 5] : (tensor<2x1x2x1x2x2xf32>) -> tensor<2x2x1x2x1x2xf32>
  // CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[TRANSPOSE0]] : (tensor<2x2x1x2x1x2xf32>) -> tensor<4x1x2x1x2xf32>
  // CHECK: %[[A2A:.*]] = "stablehlo.all_to_all"(%[[RESHAPE1]]) <{
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
  // CHECK-SAME: concat_dimension = 0 : i64,
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"z">, #sdy<axis_ref"y":(2)2>]>
  // CHECK-SAME: split_count = 4 : i64,
  // CHECK-SAME: split_dimension = 0 : i64
  // CHECK-SAME: }> : (tensor<4x1x2x1x2xf32>) -> tensor<4x1x2x1x2xf32>
  // CHECK: %[[RESHAPE2:.*]] = stablehlo.reshape %[[A2A]] : (tensor<4x1x2x1x2xf32>) -> tensor<2x2x1x2x1x2xf32>
  // CHECK: %[[TRANSPOSE1:.*]] = stablehlo.transpose %[[RESHAPE2]], dims = [2, 1, 3, 0, 4, 5] : (tensor<2x2x1x2x1x2xf32>) -> tensor<1x2x2x2x1x2xf32>
  // CHECK: %[[RESULT:.*]] = stablehlo.reshape %[[TRANSPOSE1]] : (tensor<1x2x2x2x1x2xf32>) -> tensor<1x4x2x2xf32>
  %0 = sdy.all_to_all [{"y":(2)2}: 1->3, {"z"}: 2->0] %arg0 out_sharding=<@mesh_2_4_2, [{"z"}, {"x"}, {}, {"y":(2)2}]> : tensor<2x8x2x4xf32>
  // CHECK: return %[[RESULT]] : tensor<1x4x2x2xf32>
  return %0 : tensor<2x8x2x4xf32>
}

// CHECK-LABEL: func @two_params_move_all_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x1x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {"y"}, {}]>})
// CHECK-SAME: -> (tensor<1x2x4x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}, {}, {"x"}]>}) {
func.func @two_params_move_all_axes(%arg0: tensor<4x2x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}, {"y"}, {}]>})
    -> (tensor<4x2x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}, {}, {"x"}]>}) {
  // CHECK: %[[RESHAPE0:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<4x1x1x2xf32>) -> tensor<4x1x1x1x2x1xf32>
  // CHECK: %[[TRANSPOSE0:.*]] = stablehlo.transpose %[[RESHAPE0]], dims = [0, 4, 1, 2, 3, 5] : (tensor<4x1x1x1x2x1xf32>) -> tensor<4x2x1x1x1x1xf32>
  // CHECK: %[[RESHAPE1:.*]] = stablehlo.reshape %[[TRANSPOSE0]] : (tensor<4x2x1x1x1x1xf32>) -> tensor<8x1x1x1x1xf32>
  // CHECK: %[[A2A:.*]] = "stablehlo.all_to_all"(%[[RESHAPE1]]) <{
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>,
  // CHECK-SAME: concat_dimension = 0 : i64,
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 4, 1, 5, 2, 6, 3, 7]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#sdy<axis_ref"y">, #sdy<axis_ref"x">]>
  // CHECK-SAME: split_count = 8 : i64,
  // CHECK-SAME: split_dimension = 0 : i64
  // CHECK-SAME: }> : (tensor<8x1x1x1x1xf32>) -> tensor<8x1x1x1x1xf32>
  // CHECK: %[[RESHAPE2:.*]] = stablehlo.reshape %[[A2A]] : (tensor<8x1x1x1x1xf32>) -> tensor<4x2x1x1x1x1xf32>
  // CHECK: %[[TRANSPOSE1:.*]] = stablehlo.transpose %[[RESHAPE2]], dims = [2, 1, 3, 0, 4, 5] : (tensor<4x2x1x1x1x1xf32>) -> tensor<1x2x1x4x1x1xf32>
  // CHECK: %[[RESULT:.*]] = stablehlo.reshape %[[TRANSPOSE1]] : (tensor<1x2x1x4x1x1xf32>) -> tensor<1x2x4x1xf32>
  %0 = sdy.all_to_all [{"x"}: 1->3, {"y"}: 2->0] %arg0 out_sharding=<@mesh_2_4, [{"y"}, {}, {}, {"x"}]> : tensor<4x2x4x2xf32>
  // CHECK: return %[[RESULT]] : tensor<1x2x4x1xf32>
  return %0 : tensor<4x2x4x2xf32>
}
