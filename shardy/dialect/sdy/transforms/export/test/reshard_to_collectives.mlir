// RUN: sdy_opt %s -sdy-reshard-to-collectives | FileCheck %s

sdy.mesh @mesh1d_6 = <["x"=6]>
sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh2d_4x2 = <["x"=4, "y"=2]>
sdy.mesh @mesh2d_2x8 = <["x"=2, "y"=8]>
sdy.mesh @mesh3d = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh4d_w4 = <["x"=2, "y"=2, "z"=2, "w"=4]>
sdy.mesh @mesh4d_w16 = <["x"=2, "y"=2, "z"=2, "w"=16]>

// CHECK-LABEL: func @redundant_reshard
func.func @redundant_reshard(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2d, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: return %arg0
  %0 = sdy.reshard %arg0 <@mesh2d, [{"x", ?}, {"y", ?}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard_to_all_gather_single_axis
func.func @reshard_to_all_gather_single_axis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2d, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh2d, [{"y"}, {}]>
  %0 = sdy.reshard %arg0 <@mesh2d, [{"y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard_to_all_gather_multiple_axes
func.func @reshard_to_all_gather_multiple_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3d, [{"x", "y", "z"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{"y", "z"}, {}] %arg0 out_sharding=<@mesh3d, [{"x"}, {}]>
  %0 = sdy.reshard %arg0 <@mesh3d, [{"x"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard_to_all_gather_multiple_dims
func.func @reshard_to_all_gather_multiple_dims(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3d, [{"y", "z"}, {"x"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{"z"}, {}] %arg0 out_sharding=<@mesh3d, [{"y"}, {"x"}]>
  %0 = sdy.reshard %arg0 <@mesh3d, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard_to_all_gather_with_subaxis
func.func @reshard_to_all_gather_with_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2d_2x8, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{"y":(4)2}, {}] %arg0 out_sharding=<@mesh2d_2x8, [{"y":(1)4}, {"x"}]>
 %0 = sdy.reshard %arg0 <@mesh2d_2x8, [{"y":(1)4}, {"x"}]> :  tensor<16x8xf32>
 return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard_to_all_slice_multiple_axes
func.func @reshard_to_all_slice_multiple_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3d, [{}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{"x"}, {"y", "z"}] %arg0 out_sharding=<@mesh3d, [{"x"}, {"y", "z"}]>
  %0 = sdy.reshard %arg0 <@mesh3d, [{"x"}, {"y", "z"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard_to_all_slice_minor_axis
func.func @reshard_to_all_slice_minor_axis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3d, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{}, {"z"}] %arg0 out_sharding=<@mesh3d, [{"x"}, {"y", "z"}]>
  %0 = sdy.reshard %arg0 <@mesh3d, [{"x"}, {"y", "z"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @major_axis_available_to_slice
func.func @major_axis_available_to_slice(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh4d_w4, [{"y", "z", "w"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[ALL_SLICE_0:.*]] = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh4d_w4, [{"y", "z", "w"}, {"x"}]>
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"z", "w"}, {}] %[[ALL_SLICE_0]] out_sharding=<@mesh4d_w4, [{"y"}, {"x"}]>
  // CHECK-NEXT: %[[ALL_SLICE_1:.*]] = sdy.all_slice [{}, {"w"}] %[[ALL_GATHER]] out_sharding=<@mesh4d_w4, [{"y"}, {"x", "w"}]>
  // CHECK-NEXT: return %[[ALL_SLICE_1]]
  %0 = sdy.reshard %arg0 <@mesh4d_w4, [{"y"}, {"x", "w"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @prefix_subaxis_available_to_slice
func.func @prefix_subaxis_available_to_slice(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh4d_w4, [{"y", "z", "w":(2)2}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[ALL_SLICE_0:.*]] = sdy.all_slice [{}, {"x", "w":(1)2}] %arg0 out_sharding=<@mesh4d_w4, [{"y", "z", "w":(2)2}, {"x", "w":(1)2}]>
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"z", "w":(2)2}, {}] %[[ALL_SLICE_0]] out_sharding=<@mesh4d_w4, [{"y"}, {"x", "w":(1)2}]>
  // CHECK-NEXT: %[[ALL_SLICE_1:.*]] = sdy.all_slice [{}, {"w":(2)2}] %[[ALL_GATHER]] out_sharding=<@mesh4d_w4, [{"y"}, {"x", "w"}]>
  // CHECK-NEXT: return %[[ALL_SLICE_1]]
  %0 = sdy.reshard %arg0 <@mesh4d_w4, [{"y"}, {"x", "w"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @prefix_subaxis_available_to_slice_2
func.func @prefix_subaxis_available_to_slice_2(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh4d_w16, [{"y", "w":(4)2, "z", "w":(1)2}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[ALL_SLICE_0:.*]] = sdy.all_slice [{}, {"x", "w":(2)2}] %arg0 out_sharding=<@mesh4d_w16, [{"y", "w":(4)2, "z", "w":(1)2}, {"x", "w":(2)2}]>
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"w":(4)2, "z", "w":(1)2}, {}] %[[ALL_SLICE_0]] out_sharding=<@mesh4d_w16, [{"y"}, {"x", "w":(2)2}]>
  // CHECK-NEXT: %[[ALL_SLICE_1:.*]] = sdy.all_slice [{}, {"w":(4)4}] %[[ALL_GATHER]] out_sharding=<@mesh4d_w16, [{"y"}, {"x", "w":(2)8}]>
  // CHECK-NEXT: return %[[ALL_SLICE_1]]
  %0 = sdy.reshard %arg0 <@mesh4d_w16, [{"y"}, {"x", "w":(2)8}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @split_full_axis_not_available_to_slice
func.func @split_full_axis_not_available_to_slice(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh4d_w4, [{"y", "w":(1)2, "z", "w":(2)2}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[ALL_SLICE_0:.*]] = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh4d_w4, [{"y", "w":(1)2, "z", "w":(2)2}, {"x"}]>
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"w":(1)2, "z", "w":(2)2}, {}] %[[ALL_SLICE_0]] out_sharding=<@mesh4d_w4, [{"y"}, {"x"}]>
  // CHECK-NEXT: %[[ALL_SLICE_1:.*]] = sdy.all_slice [{}, {"w"}] %[[ALL_GATHER]] out_sharding=<@mesh4d_w4, [{"y"}, {"x", "w"}]>
  // CHECK-NEXT: return %[[ALL_SLICE_1]]
  %0 = sdy.reshard %arg0 <@mesh4d_w4, [{"y"}, {"x", "w"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @prefix_subaxis_not_available_to_slice
func.func @prefix_subaxis_not_available_to_slice(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh4d_w4, [{"y", "z", "w":(1)2}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[ALL_SLICE_0:.*]] = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh4d_w4, [{"y", "z", "w":(1)2}, {"x"}]>
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"z", "w":(1)2}, {}] %[[ALL_SLICE_0]] out_sharding=<@mesh4d_w4, [{"y"}, {"x"}]>
  // CHECK-NEXT: %[[ALL_SLICE_1:.*]] = sdy.all_slice [{}, {"w"}] %[[ALL_GATHER]] out_sharding=<@mesh4d_w4, [{"y"}, {"x", "w"}]>
  // CHECK-NEXT: return %[[ALL_SLICE_1]]
  %0 = sdy.reshard %arg0 <@mesh4d_w4, [{"y"}, {"x", "w"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @prefix_and_suffix_subaxes_not_available_to_slice
func.func @prefix_and_suffix_subaxes_not_available_to_slice(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh4d_w16, [{"y", "w":(4)2, "z", "w":(1)2}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %[[ALL_SLICE_0:.*]] = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh4d_w16, [{"y", "w":(4)2, "z", "w":(1)2}, {"x"}]>
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"w":(4)2, "z", "w":(1)2}, {}] %[[ALL_SLICE_0]] out_sharding=<@mesh4d_w16, [{"y"}, {"x"}]>
  // CHECK-NEXT: %[[ALL_SLICE_1:.*]] = sdy.all_slice [{}, {"w"}] %[[ALL_GATHER]] out_sharding=<@mesh4d_w16, [{"y"}, {"x", "w"}]>
  // CHECK-NEXT: return %[[ALL_SLICE_1]]
  %0 = sdy.reshard %arg0 <@mesh4d_w16, [{"y"}, {"x", "w"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard_with_non_divisible_subaxes_same_pre_size
func.func @reshard_with_non_divisible_subaxes_same_pre_size(%arg0 : tensor<6x2xf32> {sdy.sharding=#sdy.sharding<@mesh1d_6, [{"x":(1)2}, {}]>}) -> tensor<6x2xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(1)2}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{"x":(1)3}, {}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{"x":(1)3}, {}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
 %0 = sdy.reshard %arg0 <@mesh1d_6, [{"x":(1)3}, {}]> :  tensor<6x2xf32>
 return %0 : tensor<6x2xf32>
}

// CHECK-LABEL: func @reshard_with_non_divisible_overlapping_subaxes
func.func @reshard_with_non_divisible_overlapping_subaxes(%arg0 : tensor<6x2xf32> {sdy.sharding=#sdy.sharding<@mesh1d_6, [{"x":(2)3}, {}]>}) -> tensor<6x2xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(2)3}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{"x":(1)3}, {}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{"x":(1)3}, {}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
 %0 = sdy.reshard %arg0 <@mesh1d_6, [{"x":(1)3}, {}]> :  tensor<6x2xf32>
 return %0 : tensor<6x2xf32>
}

// CHECK-LABEL: func @reshard_with_non_divisible_overlapping_diff_dim
func.func @reshard_with_non_divisible_overlapping_diff_dim(%arg0 : tensor<6x2xf32> {sdy.sharding=#sdy.sharding<@mesh1d_6, [{"x":(2)3}, {}]>}) -> tensor<6x2xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(2)3}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"x":(1)3}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{}, {"x":(1)3}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
 %0 = sdy.reshard %arg0 <@mesh1d_6, [{}, {"x":(1)3}]> :  tensor<6x2xf32>
 return %0 : tensor<6x2xf32>
}
