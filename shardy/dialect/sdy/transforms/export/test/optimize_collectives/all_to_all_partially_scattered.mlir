// RUN: sdy_opt %s -sdy-optimize-collectives | FileCheck %s

sdy.mesh @mesh_2d = <["x"=2, "y"=2]>
sdy.mesh @mesh_3d = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_subaxis_16 = <["x"=16]>
sdy.mesh @mesh_unequal_2d = <["x"=4, "y"=2]>
sdy.mesh @mesh_unequal_3d = <["x"=4, "y"=2, "z"=4]>

// 2-axis permutation where only "x" is communicated, leaving permuted "y" on
// dim 0.
// CHECK-LABEL: func @two_axis_permuted_one_scattered
// CHECK-SAME:    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<16x8xf32>) -> tensor<2x2x4x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->3] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {"y"}, {}, {"x"}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->0] %[[A2A1]] out_sharding=<@mesh_2d, [{"y"}, {}, {}, {"x"}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"y"}, {"x"}]>]>} : (tensor<2x2x4x8xf32>) -> tensor<16x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8xf32>
func.func @two_axis_permuted_one_scattered(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_2d, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// 3-axis permutation where "x" and "z" are swapped, but only "x" and "y" are
// communicated (permuted "z" stays on dim 0).
// CHECK-LABEL: func @three_axis_permuted_two_scattered
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y", "z"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"x"}, {"y"}, {"z"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->4] %[[RESHAPE_IN]] out_sharding=<@mesh_3d, [{}, {"y"}, {"z"}, {}, {"x"}, {}]> : tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->5] %[[A2A1]] out_sharding=<@mesh_3d, [{}, {}, {"z"}, {}, {"x"}, {"y"}]> : tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[A2A3:.*]] = sdy.all_to_all [{"z"}: 2->0] %[[A2A2]] out_sharding=<@mesh_3d, [{"z"}, {}, {}, {}, {"x"}, {"y"}]> : tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"z"}, {"x"}, {"y"}]>]>} : (tensor<2x2x2x2x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @three_axis_permuted_two_scattered(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y", "z"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"z"}, {"x"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"z", "y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_3d, [{"z", "y"}, {"x"}, {}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh_3d, [{"z"}, {"x"}, {"y"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// Cyclic 3-axis permutation where only 1 axis is communicated.
// CHECK-LABEL: func @cyclic_three_axis_permuted_one_scattered
// CHECK-SAME:    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y", "z"}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"x"}, {"y"}, {"z"}, {}, {}]>]>} : (tensor<16x8xf32>) -> tensor<2x2x2x2x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->4] %[[RESHAPE_IN]] out_sharding=<@mesh_3d, [{}, {"y"}, {"z"}, {}, {"x"}]> : tensor<2x2x2x2x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->0] %[[A2A1]] out_sharding=<@mesh_3d, [{"y"}, {}, {"z"}, {}, {"x"}]> : tensor<2x2x2x2x8xf32>
// CHECK:       %[[A2A3:.*]] = sdy.all_to_all [{"z"}: 2->1] %[[A2A2]] out_sharding=<@mesh_3d, [{"y"}, {"z"}, {}, {}, {"x"}]> : tensor<2x2x2x2x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"y", "z"}, {"x"}]>]>} : (tensor<2x2x2x2x8xf32>) -> tensor<16x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8xf32>
func.func @cyclic_three_axis_permuted_one_scattered(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y", "z"}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"y", "z"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"y", "z", "x"}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_3d, [{"y", "z"}, {"x"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// Non-major split dimension (dim 1) where "x" and "y" are permuted, but only
// "x" is communicated to dim 0.
// CHECK-LABEL: func @non_major_split_dim_partial_scatter
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x", "y"}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"x"}, {"y"}, {}]>]>} : (tensor<8x16xf32>) -> tensor<8x2x2x4xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 1->0] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{"x"}, {}, {"y"}, {}]> : tensor<8x2x2x4xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 2->1] %[[A2A1]] out_sharding=<@mesh_2d, [{"x"}, {"y"}, {}, {}]> : tensor<8x2x2x4xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}]>]>} : (tensor<8x2x2x4xf32>) -> tensor<8x16xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<8x16xf32>
func.func @non_major_split_dim_partial_scatter(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x", "y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{}, {"y", "x"}]> : tensor<8x16xf32>
  %1 = sdy.all_to_all [{"x"}: 1->0] %0 out_sharding=<@mesh_2d, [{"x"}, {"y"}]> : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// Tests sub-axis partial scatter where "x":(1)2 and "x":(4)2 are permuted,
// but only "x":(1)2 is communicated, leaving permuted "x":(4)2 on dim 0.
// CHECK-LABEL: func @sub_axis_partial_scatter
// CHECK-SAME:    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(1)2, "x":(4)2}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis_16, [{"x":(1)2}, {"x":(4)2}, {}, {}]>]>} : (tensor<16x8xf32>) -> tensor<2x2x4x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x":(1)2}: 0->3] %[[RESHAPE_IN]] out_sharding=<@mesh_subaxis_16, [{}, {"x":(4)2}, {}, {"x":(1)2}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"x":(4)2}: 1->0] %[[A2A1]] out_sharding=<@mesh_subaxis_16, [{"x":(4)2}, {}, {}, {"x":(1)2}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis_16, [{"x":(4)2}, {"x":(1)2}]>]>} : (tensor<2x2x4x8xf32>) -> tensor<16x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8xf32>
func.func @sub_axis_partial_scatter(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(1)2, "x":(4)2}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(4)2}, {"x":(1)2}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_subaxis_16, [{"x":(4)2, "x":(1)2}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"x":(1)2}: 0->1] %0 out_sharding=<@mesh_subaxis_16, [{"x":(4)2}, {"x":(1)2}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @untouched_axis_communicated_permuted_remain
// CHECK:       %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"z"}, {"y", "x"}]> : tensor<8x16xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"z"}: 0->1] %[[CP]] out_sharding=<@mesh_3d, [{}, {"y", "x", "z"}]> : tensor<8x16xf32>
// CHECK:       return %[[A2A]] : tensor<8x16xf32>
func.func @untouched_axis_communicated_permuted_remain(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"z"}, {"x", "y"}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{}, {"y", "x", "z"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"z"}, {"y", "x"}]> : tensor<8x16xf32>
  %1 = sdy.all_to_all [{"z"}: 0->1] %0 out_sharding=<@mesh_3d, [{}, {"y", "x", "z"}]> : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// Single multi-axis all_to_all moving {"x", "y"} from split dim 1 to dim 0
// while leaving permuted "z" on dim 1; must emit sequential single-param stages
// rather than overlapping target dim 0 in one all_to_all.
// CHECK-LABEL: func @multi_axis_single_all_to_all_on_non_major_split_dim
// CHECK-SAME:    (%arg0: tensor<7x24xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{}, {"x", "y", "z"}]>}) -> (tensor<7x24xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y"}, {"z"}]>})
// CHECK-NEXT:  %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{}, {"x"}, {"y"}, {"z"}, {}]>]>} : (tensor<7x24xf32>) -> tensor<7x2x2x2x3xf32>
// CHECK-NEXT:  %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 1->0] %[[RESHAPE_IN]] out_sharding=<@mesh_3d, [{"x"}, {}, {"y"}, {"z"}, {}]> : tensor<7x2x2x2x3xf32>
// CHECK-NEXT:  %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 2->0] %[[A2A1]] out_sharding=<@mesh_3d, [{"x", "y"}, {}, {}, {"z"}, {}]> : tensor<7x2x2x2x3xf32>
// CHECK-NEXT:  %[[A2A3:.*]] = sdy.all_to_all [{"z"}: 3->1] %[[A2A2]] out_sharding=<@mesh_3d, [{"x", "y"}, {"z"}, {}, {}, {}]> : tensor<7x2x2x2x3xf32>
// CHECK-NEXT:  %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"x", "y"}, {"z"}]>]>} : (tensor<7x2x2x2x3xf32>) -> tensor<7x24xf32>
// CHECK-NEXT:  return %[[RESHAPE_OUT]] : tensor<7x24xf32>
func.func @multi_axis_single_all_to_all_on_non_major_split_dim(%arg0: tensor<7x24xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{}, {"x", "y", "z"}]>}) -> (tensor<7x24xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y"}, {"z"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{}, {"z", "x", "y"}]> : tensor<7x24xf32>
  %1 = sdy.all_to_all [{"x", "y"}: 1->0] %0 out_sharding=<@mesh_3d, [{"x", "y"}, {"z"}]> : tensor<7x24xf32>
  return %1 : tensor<7x24xf32>
}

// Preserves collective_permute when sequential scatter of {"x", "y"} onto dim 0
// changes its divisible padded size (alignTo(6, 2) = 6 != alignTo(6, 4) = 8).
// CHECK-LABEL: func @multi_axis_single_all_to_all_uneven_padded_size_preserved
// CHECK-NEXT:  %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{}, {"z", "x", "y"}]> : tensor<6x24xf32>
// CHECK-NEXT:  %[[A2A:.*]] = sdy.all_to_all [{"x", "y"}: 1->0] %[[CP]] out_sharding=<@mesh_3d, [{"x", "y"}, {"z"}]> : tensor<6x24xf32>
// CHECK-NEXT:  return %[[A2A]] : tensor<6x24xf32>
func.func @multi_axis_single_all_to_all_uneven_padded_size_preserved(%arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{}, {"x", "y", "z"}]>}) -> (tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y"}, {"z"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{}, {"z", "x", "y"}]> : tensor<6x24xf32>
  %1 = sdy.all_to_all [{"x", "y"}: 1->0] %0 out_sharding=<@mesh_3d, [{"x", "y"}, {"z"}]> : tensor<6x24xf32>
  return %1 : tensor<6x24xf32>
}

// Partial scatter where remaining sub-axes {"x":(1)2, "x":(2)2} on split dim 0
// are merged as {"x":(1)4} in out_sharding.
// CHECK-LABEL: func @permuted_sub_axes_remaining_merged_on_split_dim
// CHECK-SAME:    (%arg0: tensor<24x6xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(8)2, "x":(2)2, "x":(1)2}, {}]>}) -> (tensor<24x6xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(1)4}, {"x":(8)2}]>})
// CHECK-NEXT:  %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis_16, [{"x":(8)2}, {"x":(2)2}, {"x":(1)2}, {}, {}]>]>} : (tensor<24x6xf32>) -> tensor<2x2x2x3x6xf32>
// CHECK-NEXT:  %[[A2A1:.*]] = sdy.all_to_all [{"x":(8)2}: 0->4] %[[RESHAPE_IN]] out_sharding=<@mesh_subaxis_16, [{}, {"x":(2)2}, {"x":(1)2}, {}, {"x":(8)2}]> : tensor<2x2x2x3x6xf32>
// CHECK-NEXT:  %[[A2A2:.*]] = sdy.all_to_all [{"x":(1)2}: 2->0] %[[A2A1]] out_sharding=<@mesh_subaxis_16, [{"x":(1)2}, {"x":(2)2}, {}, {}, {"x":(8)2}]> : tensor<2x2x2x3x6xf32>
// CHECK-NEXT:  %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis_16, [{"x":(1)4}, {"x":(8)2}]>]>} : (tensor<2x2x2x3x6xf32>) -> tensor<24x6xf32>
// CHECK-NEXT:  return %[[RESHAPE_OUT]] : tensor<24x6xf32>
func.func @permuted_sub_axes_remaining_merged_on_split_dim(%arg0: tensor<24x6xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(8)2, "x":(2)2, "x":(1)2}, {}]>}) -> (tensor<24x6xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(1)4}, {"x":(8)2}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_subaxis_16, [{"x":(1)4, "x":(8)2}, {}]> : tensor<24x6xf32>
  %1 = sdy.all_to_all [{"x":(8)2}: 0->1] %0 out_sharding=<@mesh_subaxis_16, [{"x":(1)4}, {"x":(8)2}]> : tensor<24x6xf32>
  return %1 : tensor<24x6xf32>
}

// Single remaining axis ("y"=2) moved into leading sub-dim of size 4 (from
// "x"=4); divisible so residual factor 2 merges into the unsharded tail.
// CHECK-LABEL: func @unequal_axes_single_remaining_divisible_sub_dim
// CHECK-SAME:    (%arg0: tensor<16x12xf32> {sdy.sharding = #sdy.sharding<@mesh_unequal_2d, [{"x", "y"}, {}]>}) -> (tensor<16x12xf32> {sdy.sharding = #sdy.sharding<@mesh_unequal_2d, [{"y"}, {"x"}]>})
// CHECK-NEXT:  %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_unequal_2d, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<16x12xf32>) -> tensor<4x2x2x12xf32>
// CHECK-NEXT:  %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->3] %[[RESHAPE_IN]] out_sharding=<@mesh_unequal_2d, [{}, {"y"}, {}, {"x"}]> : tensor<4x2x2x12xf32>
// CHECK-NEXT:  %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->0] %[[A2A1]] out_sharding=<@mesh_unequal_2d, [{"y"}, {}, {}, {"x"}]> : tensor<4x2x2x12xf32>
// CHECK-NEXT:  %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_unequal_2d, [{"y"}, {"x"}]>]>} : (tensor<4x2x2x12xf32>) -> tensor<16x12xf32>
// CHECK-NEXT:  return %[[RESHAPE_OUT]] : tensor<16x12xf32>
func.func @unequal_axes_single_remaining_divisible_sub_dim(%arg0: tensor<16x12xf32> {sdy.sharding = #sdy.sharding<@mesh_unequal_2d, [{"x", "y"}, {}]>}) -> (tensor<16x12xf32> {sdy.sharding = #sdy.sharding<@mesh_unequal_2d, [{"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_unequal_2d, [{"y", "x"}, {}]> : tensor<16x12xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_unequal_2d, [{"y"}, {"x"}]> : tensor<16x12xf32>
  return %1 : tensor<16x12xf32>
}

// Remaining axes {"y"=2, "z"=4} cannot move into decomposed sub-dims [4, 2]
// (non-last sub-dim 0 has size 4 != 2 and sub-dim 1 has non-divisible size 2 < 4);
// collective_permute must be preserved.
// CHECK-LABEL: func @unequal_axes_remaining_incompatible_sub_dim_preserved
// CHECK-NEXT:  %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_unequal_3d, [{"y", "z", "x"}, {}]> : tensor<32x12xf32>
// CHECK-NEXT:  %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->1] %[[CP]] out_sharding=<@mesh_unequal_3d, [{"y", "z"}, {"x"}]> : tensor<32x12xf32>
// CHECK-NEXT:  return %[[A2A]] : tensor<32x12xf32>
func.func @unequal_axes_remaining_incompatible_sub_dim_preserved(%arg0: tensor<32x12xf32> {sdy.sharding = #sdy.sharding<@mesh_unequal_3d, [{"x", "y", "z"}, {}]>}) -> (tensor<32x12xf32> {sdy.sharding = #sdy.sharding<@mesh_unequal_3d, [{"y", "z"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_unequal_3d, [{"y", "z", "x"}, {}]> : tensor<32x12xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_unequal_3d, [{"y", "z"}, {"x"}]> : tensor<32x12xf32>
  return %1 : tensor<32x12xf32>
}
