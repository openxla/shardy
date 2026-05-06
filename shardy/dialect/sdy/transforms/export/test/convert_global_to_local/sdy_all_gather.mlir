// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,COMBINED,V1
// RUN: sdy_opt %s -sdy-convert-global-to-local='per-dim-all-gather=true' | FileCheck %s --check-prefixes=CHECK,PER-DIM,V1

// RUN: sdy_opt %s -sdy-convert-global-to-local='enable-rgv3=true' | FileCheck %s --check-prefixes=CHECK,COMBINED,V3
// RUN: sdy_opt %s -sdy-convert-global-to-local='per-dim-all-gather=true enable-rgv3=true' | FileCheck %s --check-prefixes=CHECK,PER-DIM,V3

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// CHECK-LABEL: func @one_dim
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
// CHECK-SAME:    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
func.func @one_dim(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}){
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // V1-SAME{LITERAL}:   replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]>
  // V3-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#sdy<axis_ref"y">]>
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x16xf32>) -> tensor<4x16xf32>
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_2_4, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[GATHER]] : tensor<4x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @one_dim_two_axes_xy
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z"}]>})
func.func @one_dim_two_axes_xy(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z"}]>}){
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
  // V1-SAME{LITERAL}:   replica_groups = dense<[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]>
  // V3-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"x">, #sdy<axis_ref"y">]>
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.all_gather [{"x", "y"}, {}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"z"}]> : tensor<8x16xf32>
  // CHECK: return %[[GATHER]] : tensor<8x8xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @one_dim_two_axes_yx
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y", "x"}, {"z"}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z"}]>})
func.func @one_dim_two_axes_yx(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"y", "x"}, {"z"}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z"}]>}){
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
  // V1-SAME{LITERAL}:   replica_groups = dense<[[0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15]]>
  // V3-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"y">, #sdy<axis_ref"x">]>
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.all_gather [{"y", "x"}, {}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"z"}]> : tensor<8x16xf32>
  // CHECK: return %[[GATHER]] : tensor<8x8xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @one_dim_two_axes_subaxis
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y":(2)2}, {"z"}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z"}]>})
func.func @one_dim_two_axes_subaxis(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y":(2)2}, {"z"}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z"}]>}){
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>,
  // V1-SAME{LITERAL}:   replica_groups = dense<[[0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15]]>
  // V3-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"x">, #sdy<axis_ref"y":(2)2>]>
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<2x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.all_gather [{"x", "y":(2)2}, {}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"z"}]> : tensor<8x16xf32>
  // CHECK: return %[[GATHER]] : tensor<8x8xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @two_dims_yz
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
// CHECK-SAME: -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {}]>})
func.func @two_dims_yz(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {}]>}){

  // --- Per-Dimension All-Gather Strategy (per-dim-all-gather=true) ---
  // PER-DIM: %[[GATHER1:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // PER-DIM-SAME: all_gather_dim = 1 : i64,
  // PER-DIM-SAME: channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>,
  // PER-DIM-V1-SAME{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]>
  // PER-DIM-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"z">]>, use_global_device_ids}>
  // PER-DIM-SAME: }> : (tensor<1x8xf32>) -> tensor<1x16xf32>
  // PER-DIM: %[[RESULT:.*]] = "stablehlo.all_gather"(%[[GATHER1]]) <{
  // PER-DIM-SAME: all_gather_dim = 0 : i64,
  // PER-DIM-SAME: channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>,
  // PER-DIM-V1-SAME{LITERAL}: replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]>
  // PER-DIM-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"y">]>, use_global_device_ids}>
  // PER-DIM-SAME: }> : (tensor<1x16xf32>) -> tensor<4x16xf32>

  // --- Combined Dimensions All-Gather Strategy (per-dim-all-gather=false) ---
  // COMBINED: %[[RESHAPE1:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
  // COMBINED: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[RESHAPE1]]) <{
  // COMBINED-SAME: all_gather_dim = 0 : i64,
  // COMBINED-SAME: channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>,
  // COMBINED-V1-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]>
  // COMBINED-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"y">, #sdy<axis_ref"z">]>
  // COMBINED-SAME: }> : (tensor<1x1x8xf32>) -> tensor<8x1x8xf32>
  // COMBINED: %[[RESHAPE2:.*]] = stablehlo.reshape %[[GATHER]] : (tensor<8x1x8xf32>) -> tensor<4x2x1x8xf32>
  // COMBINED: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RESHAPE2]], dims = [0, 2, 1, 3] : (tensor<4x2x1x8xf32>) -> tensor<4x1x2x8xf32>
  // COMBINED: %[[RESULT:.*]] = stablehlo.reshape %[[TRANSPOSE]] : (tensor<4x1x2x8xf32>) -> tensor<4x16xf32>

  %0 = sdy.all_gather[{"y"}, {"z"}] %arg0 out_sharding = <@mesh_2_4_2, [{"x"}, {}]> : tensor<8x16xf32>

  // CHECK: return %[[RESULT]] : tensor<4x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @two_dims_yx
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"z", "y"}, {"x"}]>})
// CHECK-SAME: -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"z"}, {}]>})
func.func @two_dims_yx(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"z", "y"}, {"x"}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"z"}, {}]>}){

  // --- Per-Dimension All-Gather Strategy (per-dim-all-gather=true) ---
  // PER-DIM: %[[GATHER1:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // PER-DIM-SAME: all_gather_dim = 1 : i64,
  // PER-DIM-SAME: channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>,
  // PER-DIM-V1-SAME{LITERAL}: replica_groups = dense<[[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]>
  // PER-DIM-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"x">]>, use_global_device_ids}>
  // PER-DIM-SAME: }> : (tensor<1x8xf32>) -> tensor<1x16xf32>
  // PER-DIM: %[[RESULT:.*]] = "stablehlo.all_gather"(%[[GATHER1]]) <{
  // PER-DIM-SAME: all_gather_dim = 0 : i64,
  // PER-DIM-SAME: channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>,
  // PER-DIM-V1-SAME{LITERAL}: replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]>
  // PER-DIM-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"y">]>, use_global_device_ids}>
  // PER-DIM-SAME: use_global_device_ids
  // PER-DIM-SAME: }> : (tensor<1x16xf32>) -> tensor<4x16xf32>

  // --- Combined Dimensions All-Gather Strategy (per-dim-all-gather=false) ---
  // COMBINED: %[[RESHAPE1:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
  // COMBINED: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[RESHAPE1]]) <{
  // COMBINED-SAME: all_gather_dim = 0 : i64,
  // COMBINED-SAME: channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>,
  // COMBINED-V1-SAME{LITERAL}: replica_groups = dense<[[0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15]]>
  // COMBINED-V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#sdy<axis_ref"y">, #sdy<axis_ref"x">]>
  // COMBINED-SAME: }> : (tensor<1x1x8xf32>) -> tensor<8x1x8xf32>
  // COMBINED: %[[RESHAPE2:.*]] = stablehlo.reshape %[[GATHER]] : (tensor<8x1x8xf32>) -> tensor<4x2x1x8xf32>
  // COMBINED: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RESHAPE2]], dims = [0, 2, 1, 3] : (tensor<4x2x1x8xf32>) -> tensor<4x1x2x8xf32>
  // COMBINED: %[[RESULT:.*]] = stablehlo.reshape %[[TRANSPOSE]] : (tensor<4x1x2x8xf32>) -> tensor<4x16xf32>

  %0 = sdy.all_gather[{"y"}, {"x"}] %arg0 out_sharding = <@mesh_2_4_2, [{"z"}, {}]> : tensor<8x16xf32>

  // CHECK: return %[[RESULT]] : tensor<4x16xf32>
  return %0 : tensor<8x16xf32>
}
