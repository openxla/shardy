// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,CHECK-AR-DS
// RUN: sdy_opt %s -sdy-convert-global-to-local='combine-multi-dimensions-reduce-scatter=true' | FileCheck %s --check-prefixes=CHECK,CHECK-COMBINED

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
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
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
func.func @one_dim_sharded(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2_4, [{"x", "y":(2)2}, {}]>}) {
  // CHECK: %[[RES:.*]] = "stablehlo.reduce_scatter"(%[[ARG0]]) <{
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>
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
func.func @two_dim_add_suffix_of_full(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2_4_2, [{"y":(1)2}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2_4_2, [{"y", "z"}, {"x"}]>}) {
  // --- Use All-Reduce and Dynamic Slice (combine-multi-dimensions-reduce-scatter=false) ---
  // CHECK-AR-DS-NEXT: %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%[[ARG0]]) <{
  // CHECK-AR-DS-SAME: channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>
  // CHECK-AR-DS-SAME{LITERAL}: replica_groups = dense<[[0, 8, 1, 9, 2, 10, 3, 11], [4, 12, 5, 13, 6, 14, 7, 15]]> : tensor<2x8xi64>
  // CHECK-AR-DS-SAME: use_global_device_ids
  // CHECK-AR-DS-SAME: }> ({
  // CHECK-AR-DS-NEXT: ^bb0(%[[RS_ARG1:.*]]: tensor<f32>, %[[RS_ARG2:.*]]: tensor<f32>):
  // CHECK-AR-DS-NEXT:   %[[ADD:.*]] = stablehlo.add %[[RS_ARG1]], %[[RS_ARG2]] : tensor<f32>
  // CHECK-AR-DS-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-AR-DS: }) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  //
  // CHECK-AR-DS: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-AR-DS: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  //
  // CHECK-AR-DS: %[[TABLE0:.*]] = stablehlo.constant dense<[0, 2, 0, 2, 4, 6, 4, 6, 0, 2, 0, 2, 4, 6, 4, 6]> : tensor<16xi64>
  // CHECK-AR-DS: %[[OFF0:.*]] = stablehlo.dynamic_slice %[[TABLE0]], %[[PID_I64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK-AR-DS: %[[IDX0:.*]] = stablehlo.reshape %[[OFF0]] : (tensor<1xi64>) -> tensor<i64>
  //
  // CHECK-AR-DS: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-AR-DS: %[[PID_I64_1:.*]] = stablehlo.convert %[[PID1]] : (tensor<ui32>) -> tensor<i64>
  //
  // CHECK-AR-DS: %[[TABLE1:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4]> : tensor<16xi64>
  // CHECK-AR-DS: %[[OFF1:.*]] = stablehlo.dynamic_slice %[[TABLE1]], %[[PID_I64_1]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK-AR-DS: %[[IDX1:.*]] = stablehlo.reshape %[[OFF1]] : (tensor<1xi64>) -> tensor<i64>
  //
  // CHECK-AR-DS: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ALL_REDUCE]], %[[IDX0]], %[[IDX1]], sizes = [2, 4] : (tensor<8x8xf32>, tensor<i64>, tensor<i64>) -> tensor<2x4xf32>

  // --- Combine Multi-Dimensions Reduce Scatter (combine-multi-dimensions-reduce-scatter=true) ---
  // CHECK-COMBINED-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<8x8xf32>) -> tensor<4x2x2x4xf32>
  // CHECK-COMBINED-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RESHAPE0]], dims = [0, 2, 1, 3] : (tensor<4x2x2x4xf32>) -> tensor<4x2x2x4xf32>
  // CHECK-COMBINED-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[TRANSPOSE]] : (tensor<4x2x2x4xf32>) -> tensor<8x2x4xf32>
  //
  // CHECK-COMBINED-NEXT: %[[RS:.*]] = "stablehlo.reduce_scatter"(%[[RESHAPE1]]) <{
  // CHECK-COMBINED-SAME:   channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
  // CHECK-COMBINED-SAME{LITERAL}: replica_groups = dense<[[0, 8, 1, 9, 2, 10, 3, 11], [4, 12, 5, 13, 6, 14, 7, 15]]> : tensor<2x8xi64>,
  // CHECK-COMBINED-SAME:   scatter_dimension = 0 : i64,
  // CHECK-COMBINED-SAME:   use_global_device_ids
  // CHECK-COMBINED-SAME: }> ({
  // CHECK-COMBINED-NEXT: ^bb0(%[[RED_ARG1:.*]]: tensor<f32>, %[[RED_ARG2:.*]]: tensor<f32>):
  // CHECK-COMBINED-NEXT:   %[[SUM:.*]] = stablehlo.add %[[RED_ARG1]], %[[RED_ARG2]] : tensor<f32>
  // CHECK-COMBINED-NEXT:   stablehlo.return %[[SUM]] : tensor<f32>
  // CHECK-COMBINED-NEXT: }) : (tensor<8x2x4xf32>) -> tensor<1x2x4xf32>
  //
  // CHECK-COMBINED-NEXT: %[[RESULT:.*]] = stablehlo.reshape %[[RS]] : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
  %0 = sdy.reduce_scatter [{"y":(2)2, "z"}, {"x"}] %arg0 out_sharding=<@mesh_2_4_2, [{"y", "z"}, {"x"}]> : tensor<16x8xf32>
  // CHECK: return %[[RESULT]] : tensor<2x4xf32>
  return %0 : tensor<16x8xf32>
}
