// RUN: sdy_opt %s -sdy-convert-global-to-local='enable-rgv3=false' | FileCheck %s --check-prefixes=CHECK,AR-DS,V1,AR-DS-V1
// RUN: sdy_opt %s -sdy-convert-global-to-local='combine-multi-dimension-reduce-scatter=true enable-rgv3=false' | FileCheck %s --check-prefixes=CHECK,COMBINED,V1,COMBINED-V1

// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,AR-DS,V3,AR-DS-V3
// RUN: sdy_opt %s -sdy-convert-global-to-local='combine-multi-dimension-reduce-scatter=true' | FileCheck %s --check-prefixes=CHECK,COMBINED,V3,COMBINED-V3

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// CHECK-LABEL: func @one_dim_not_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
func.func @one_dim_not_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {"x"}]>}) {
  // CHECK: %[[RES:.*]] = "stablehlo.reduce_scatter"(%[[ARG0]])
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "x">]>
  // CHECK-SAME: scatter_dimension = 1 : i64
  // CHECK-SAME: use_global_device_ids
  // CHECK: (%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK:   %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
  // CHECK:   stablehlo.return %1 : tensor<f32>
  // CHECK: (tensor<2x8xf32>) -> tensor<2x4xf32>
  %0 = sdy.reduce_scatter [{}, {"x"}] %arg0 out_sharding=<@mesh_2_4, [{"y"}, {"x"}]> : tensor<8x8xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @one_dim_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y":(2)2}, {}]>})
func.func @one_dim_sharded(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y":(2)2}, {}]>}) {
  // CHECK: %[[RES:.*]] = "stablehlo.reduce_scatter"(%[[ARG0]]) <{
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y", sub_axis_info = (2)2>]>
  // CHECK-SAME: scatter_dimension = 0 : i64
  // CHECK-SAME: use_global_device_ids
  // CHECK-SAME: }> ({
  // CHECK: ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK:   %[[SUM:.*]] = stablehlo.add %arg1, %arg2 : tensor<f32>
  // CHECK:   stablehlo.return %[[SUM]] : tensor<f32>
  // CHECK: }) : (tensor<8x8xf32>) -> tensor<4x8xf32>
  %0 = sdy.reduce_scatter [{"y":(2)2}, {}] %arg0 out_sharding=<@mesh_2_4, [{"x", "y":(2)2}, {}]> : tensor<16x8xf32>
  // CHECK: return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @two_dim_add_suffix_of_full
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y":(1)2}, {}]>})
// CHECK-SAME: -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y", "z"}, {"x"}]>})
func.func @two_dim_add_suffix_of_full(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y":(1)2}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y", "z"}, {"x"}]>}) {
  // --- Use All-Reduce and Dynamic Slice (combine-multi-dimension-reduce-scatter=false) ---
  // AR-DS-NEXT: %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%[[ARG0]]) <{
  // AR-DS-SAME: channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>
  // AR-DS-V1-SAME{LITERAL}: replica_groups = dense<[[0, 8, 1, 9, 2, 10, 3, 11], [4, 12, 5, 13, 6, 14, 7, 15]]>
  // AR-DS-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#stablehlo.axis_ref<name = "y", sub_axis_info = (2)2>, #stablehlo.axis_ref<name = "z">, #stablehlo.axis_ref<name = "x">]>
  // AR-DS-SAME: use_global_device_ids
  // AR-DS-SAME: }> ({
  // AR-DS-NEXT: ^bb0(%[[RS_ARG1:.*]]: tensor<f32>, %[[RS_ARG2:.*]]: tensor<f32>):
  // AR-DS-NEXT:   %[[ADD:.*]] = stablehlo.add %[[RS_ARG1]], %[[RS_ARG2]] : tensor<f32>
  // AR-DS-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // AR-DS: }) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  //
  // AR-DS: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // AR-DS: %[[C2_0:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // AR-DS: %[[DIV0:.*]] = stablehlo.divide %[[PID0]], %[[C2_0]] : tensor<ui32>
  // AR-DS: %[[C2_1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // AR-DS: %[[REM0:.*]] = stablehlo.remainder %[[DIV0]], %[[C2_1]] : tensor<ui32>
  // AR-DS: %[[C2_3:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // AR-DS: %[[REM1:.*]] = stablehlo.remainder %[[PID0]], %[[C2_3]] : tensor<ui32>
  // AR-DS: %[[C2_2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // AR-DS: %[[MUL0:.*]] = stablehlo.multiply %[[REM0]], %[[C2_2]] : tensor<ui32>
  // AR-DS: %[[ADD0:.*]] = stablehlo.add %[[MUL0]], %[[REM1]] : tensor<ui32>
  // AR-DS: %[[CONV0:.*]] = stablehlo.convert %[[ADD0]] : (tensor<ui32>) -> tensor<i32>
  // AR-DS: %[[C2_I32:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // AR-DS: %[[IDX0:.*]] = stablehlo.multiply %[[CONV0]], %[[C2_I32]] : tensor<i32>
  //
  // AR-DS: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // AR-DS: %[[C8_UI32:.*]] = stablehlo.constant dense<8> : tensor<ui32>
  // AR-DS: %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C8_UI32]] : tensor<ui32>
  // AR-DS: %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // AR-DS: %[[C4_I32:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // AR-DS: %[[IDX1:.*]] = stablehlo.multiply %[[CONV1]], %[[C4_I32]] : tensor<i32>
  //
  // AR-DS: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ALL_REDUCE]], %[[IDX0]], %[[IDX1]], sizes = [2, 4] : (tensor<8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<2x4xf32>

  // --- Combine Multi-Dimension Reduce Scatter (combine-multi-dimension-reduce-scatter=true) ---
  // COMBINED-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<8x8xf32>) -> tensor<4x2x2x4xf32>
  // COMBINED-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RESHAPE0]], dims = [0, 2, 1, 3] : (tensor<4x2x2x4xf32>) -> tensor<4x2x2x4xf32>
  // COMBINED-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[TRANSPOSE]] : (tensor<4x2x2x4xf32>) -> tensor<8x2x4xf32>
  //
  // COMBINED-NEXT: %[[RS:.*]] = "stablehlo.reduce_scatter"(%[[RESHAPE1]]) <{
  // COMBINED-SAME:   channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
  // COMBINED-V1-SAME{LITERAL}: replica_groups = dense<[[0, 8, 1, 9, 2, 10, 3, 11], [4, 12, 5, 13, 6, 14, 7, 15]]>
  // COMBINED-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#stablehlo.axis_ref<name = "y", sub_axis_info = (2)2>, #stablehlo.axis_ref<name = "z">, #stablehlo.axis_ref<name = "x">]>
  // COMBINED-SAME:   scatter_dimension = 0 : i64,
  // COMBINED-SAME:   use_global_device_ids
  // COMBINED-SAME: }> ({
  // COMBINED-NEXT: ^bb0(%[[RED_ARG1:.*]]: tensor<f32>, %[[RED_ARG2:.*]]: tensor<f32>):
  // COMBINED-NEXT:   %[[SUM:.*]] = stablehlo.add %[[RED_ARG1]], %[[RED_ARG2]] : tensor<f32>
  // COMBINED-NEXT:   stablehlo.return %[[SUM]] : tensor<f32>
  // COMBINED-NEXT: }) : (tensor<8x2x4xf32>) -> tensor<1x2x4xf32>
  //
  // COMBINED-NEXT: %[[RESULT:.*]] = stablehlo.reshape %[[RS]] : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
  %0 = sdy.reduce_scatter [{"y":(2)2, "z"}, {"x"}] %arg0 out_sharding=<@mesh_2_4_2, [{"y", "z"}, {"x"}]> : tensor<16x8xf32>
  // CHECK: return %[[RESULT]] : tensor<2x4xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_max
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
func.func @reduce_scatter_max(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {"x"}]>}) {
  // CHECK: %[[RES:.*]] = "stablehlo.reduce_scatter"(%[[ARG0]])
  // CHECK-SAME: scatter_dimension = 1 : i64
  // CHECK-SAME: use_global_device_ids
  // CHECK: (%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK:   %[[MAX:.*]] = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  // CHECK:   stablehlo.return %[[MAX]] : tensor<f32>
  // CHECK: (tensor<2x8xf32>) -> tensor<2x4xf32>
  %0 = sdy.reduce_scatter max [{}, {"x"}] %arg0 out_sharding=<@mesh_2_4, [{"y"}, {"x"}]> : tensor<8x8xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_min
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
func.func @reduce_scatter_min(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"y"}, {"x"}]>}) {
  // CHECK: %[[RES:.*]] = "stablehlo.reduce_scatter"(%[[ARG0]])
  // CHECK-SAME: scatter_dimension = 1 : i64
  // CHECK-SAME: use_global_device_ids
  // CHECK: (%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK:   %[[MIN:.*]] = stablehlo.minimum %arg1, %arg2 : tensor<f32>
  // CHECK:   stablehlo.return %[[MIN]] : tensor<f32>
  // CHECK: (tensor<2x8xf32>) -> tensor<2x4xf32>
  %0 = sdy.reduce_scatter min [{}, {"x"}] %arg0 out_sharding=<@mesh_2_4, [{"y"}, {"x"}]> : tensor<8x8xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  return %0 : tensor<8x8xf32>
}
