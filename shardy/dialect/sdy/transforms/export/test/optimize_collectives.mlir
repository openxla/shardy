// RUN: sdy_opt %s -sdy-optimize-collectives | FileCheck %s

sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh2d_2x8 = <["x"=2, "y"=8]>
sdy.mesh @mesh3d = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_spatial = <["batch"=2, "shard"=8]>

// CHECK-LABEL: func @input_dim_split_and_swapped
func.func @input_dim_split_and_swapped(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{"x", "y"}, {}, {}]>}) -> tensor<8x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<8x8x8xf32>) -> tensor<2x4x8x8xf32>
  // CHECK-NEXT: %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->2] %[[RESHAPE_0]] out_sharding=<@mesh2d, [{}, {"y"}, {"x"}, {}]> : tensor<2x4x8x8xf32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[A2A]], dims = [1, 0, 2, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d, [{"y"}, {}, {"x"}, {}]>]>} : (tensor<2x4x8x8xf32>) -> tensor<4x2x8x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[TRANSPOSE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d, [{"y"}, {"x"}, {}]>]>} : (tensor<4x2x8x8xf32>) -> tensor<8x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh2d, [{"y", "x"}, {}, {}]> : tensor<8x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh2d, [{"y"}, {"x"}, {}]> : tensor<8x8x8xf32>
  return %1 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func @input_dim_split_into_three_groups
func.func @input_dim_split_into_three_groups(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"x", "y", "z"}, {}, {}]>}) -> tensor<16x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"x"}, {"y"}, {"z"}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
  // CHECK-NEXT: %[[A2A_1:.*]] = sdy.all_to_all [{"z"}: 2->4] %[[RESHAPE_0]] out_sharding=<@mesh3d, [{"x"}, {"y"}, {}, {}, {"z"}]> : tensor<2x2x4x8x8xf32>
  // CHECK-NEXT: %[[A2A_2:.*]] = sdy.all_to_all [{"x"}: 0->3] %[[A2A_1]] out_sharding=<@mesh3d, [{}, {"y"}, {}, {"x"}, {"z"}]> : tensor<2x2x4x8x8xf32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[A2A_2]], dims = [1, 0, 2, 3, 4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"y"}, {}, {}, {"x"}, {"z"}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<2x2x4x8x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[TRANSPOSE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"y"}, {"x"}, {"z"}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh3d, [{"y", "x", "z"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"z"}: 0->2] %0 out_sharding=<@mesh3d, [{"y", "x"}, {}, {"z"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"x"}: 0->1] %1 out_sharding=<@mesh3d, [{"y"}, {"x"}, {"z"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @input_dim_split_with_subaxes
func.func @input_dim_split_with_subaxes(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d_2x8, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d_2x8, [{"y":(1)2}, {"y":(2)4}, {}]>]>} : (tensor<16x8xf32>) -> tensor<2x8x8xf32>
  // CHECK-NEXT: %[[A2A:.*]] = sdy.all_to_all [{"y":(1)2}: 0->2] %[[RESHAPE_0]] out_sharding=<@mesh2d_2x8, [{}, {"y":(2)4}, {"y":(1)2}]> : tensor<2x8x8xf32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[A2A]], dims = [1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d_2x8, [{"y":(2)4}, {}, {"y":(1)2}]>]>} : (tensor<2x8x8xf32>) -> tensor<8x2x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[TRANSPOSE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d_2x8, [{"y":(2)4}, {"y":(1)2}]>]>} : (tensor<8x2x8xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh2d_2x8, [{"y":(2)4, "y":(1)2}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"y":(1)2}: 0->1] %0 out_sharding=<@mesh2d_2x8, [{"y":(2)4}, {"y":(1)2}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @no_cp_no_split
func.func @no_cp_no_split(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d_2x8, [{"y"}, {}, {}]>}) -> tensor<16x8x8xf32> {
  // CHECK-NEXT: %[[A2A_0:.*]] = sdy.all_to_all [{"y":(2)4}: 0->2] %arg0 out_sharding=<@mesh2d_2x8, [{"y":(1)2}, {}, {"y":(2)4}]> : tensor<16x8x8xf32>
  // CHECK-NEXT: %[[A2A_1:.*]] = sdy.all_to_all [{"y":(1)2}: 0->1] %[[A2A_0]] out_sharding=<@mesh2d_2x8, [{}, {"y":(1)2}, {"y":(2)4}]> : tensor<16x8x8xf32>
  // CHECK-NEXT: return %[[A2A_1]]
  %0 = sdy.all_to_all [{"y":(2)4}: 0->2] %arg0 out_sharding=<@mesh2d_2x8, [{"y":(1)2}, {}, {"y":(2)4}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"y":(1)2}: 0->1] %0 out_sharding=<@mesh2d_2x8, [{}, {"y":(1)2}, {"y":(2)4}]> : tensor<16x8x8xf32>
  return %1 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @input_dim_split_mismatched_strides
func.func @input_dim_split_mismatched_strides(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"x", "y", "z"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh3d, [{"z", "y", "x"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->1] %[[CP]] out_sharding=<@mesh3d, [{"z", "y"}, {"x"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %[[A2A]]
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh3d, [{"z", "y", "x"}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh3d, [{"z", "y"}, {"x"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @spatial_partitioning_batch_shard
func.func @spatial_partitioning_batch_shard(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_spatial, [{"batch", "shard"}, {}, {}, {}]>}) -> tensor<16x8x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_spatial, [{"batch"}, {"shard"}, {}, {}, {}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<2x8x8x8x8xf32>
  // CHECK-NEXT: %[[A2A_1:.*]] = sdy.all_to_all [{"batch"}: 0->2] %[[RESHAPE_0]] out_sharding=<@mesh_spatial, [{}, {"shard"}, {"batch"}, {}, {}]> : tensor<2x8x8x8x8xf32>
  // CHECK-NEXT: %[[A2A_2:.*]] = sdy.all_to_all [{"shard"}: 1->3] %[[A2A_1]] out_sharding=<@mesh_spatial, [{}, {}, {"batch"}, {"shard"}, {}]> : tensor<2x8x8x8x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[A2A_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_spatial, [{}, {"batch"}, {"shard"}, {}]>]>} : (tensor<2x8x8x8x8xf32>) -> tensor<16x8x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_spatial, [{"shard", "batch"}, {}, {}, {}]> : tensor<16x8x8x8xf32>
  %1 = sdy.all_to_all [{"batch"}: 0->1] %0 out_sharding=<@mesh_spatial, [{"shard"}, {"batch"}, {}, {}]> : tensor<16x8x8x8xf32>
  %2 = sdy.all_to_all [{"shard"}: 0->2] %1 out_sharding=<@mesh_spatial, [{}, {"batch"}, {"shard"}, {}]> : tensor<16x8x8x8xf32>
  return %2 : tensor<16x8x8x8xf32>
}
