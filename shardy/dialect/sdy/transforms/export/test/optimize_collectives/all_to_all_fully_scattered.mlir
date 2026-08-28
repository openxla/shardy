// RUN: sdy_opt %s -sdy-optimize-collectives | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_2d = <["x"=2, "y"=2]>
sdy.mesh @mesh_sub = <["x"=4, "y"=2]>

// Tests 2-axis full scatter on split dimension 0 where both axes {"x", "y"}
// are permuted and communicated to separate target dimensions.
// CHECK-LABEL: func @two_axis_full_scatter
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->4] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {"y"}, {}, {}, {"x"}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->3] %[[A2A1]] out_sharding=<@mesh_2d, [{}, {}, {}, {"y"}, {"x"}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"y"}, {"x"}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @two_axis_full_scatter(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// Tests 2-axis scatter where both axes {"x", "y"} on split dimension 0 are
// permuted and communicated to the same target dimension (dimension 1) across
// sequential all-to-all operations.
// CHECK-LABEL: func @two_axis_scatter_to_same_target_dim
// CHECK-SAME:    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<16x8xf32>) -> tensor<2x2x4x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->3] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {"y"}, {}, {"x"}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->3] %[[A2A1]] out_sharding=<@mesh_2d, [{}, {}, {}, {"x", "y"}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"x", "y"}]>]>} : (tensor<2x2x4x8xf32>) -> tensor<16x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8xf32>
func.func @two_axis_scatter_to_same_target_dim(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x", "y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_2d, [{"y"}, {"x"}]> : tensor<16x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"x", "y"}]> : tensor<16x8xf32>
  return %2 : tensor<16x8xf32>
}

// Tests 3-axis full scatter on split dimension 0 where all three axes
// {"x", "y", "z"} are permuted and communicated to separate target dimensions.
// CHECK-LABEL: func @three_axis_full_scatter
// CHECK-SAME:    %arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y", "z"}, {}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {"z"}, {}, {}, {}, {}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->6] %[[RESHAPE_IN]] out_sharding=<@mesh, [{}, {"y"}, {"z"}, {}, {}, {}, {"x"}]> : tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->5] %[[A2A1]] out_sharding=<@mesh, [{}, {}, {"z"}, {}, {}, {"y"}, {"x"}]> : tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[A2A3:.*]] = sdy.all_to_all [{"z"}: 2->4] %[[A2A2]] out_sharding=<@mesh, [{}, {}, {}, {}, {"z"}, {"y"}, {"x"}]> : tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}, {"y"}, {"x"}]>]>} : (tensor<2x2x2x2x8x8x8xf32>) -> tensor<16x8x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8x8xf32>
func.func @three_axis_full_scatter(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y", "z"}, {}, {}, {}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"z"}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"z", "y", "x"}, {}, {}, {}]> : tensor<16x8x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->3] %0 out_sharding=<@mesh, [{"z", "y"}, {}, {}, {"x"}]> : tensor<16x8x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh, [{"z"}, {}, {"y"}, {"x"}]> : tensor<16x8x8x8xf32>
  %3 = sdy.all_to_all [{"z"}: 0->1] %2 out_sharding=<@mesh, [{}, {"z"}, {"y"}, {"x"}]> : tensor<16x8x8x8xf32>
  return %3 : tensor<16x8x8x8xf32>
}

// Tests 3-axis sharding where unpermuted axis "z" stays on split dimension 0
// while permuted axes "x" and "y" are communicated to separate target
// dimensions.
// CHECK-LABEL: func @two_axis_scatter_with_untouched_axis
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z", "x", "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"x"}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 1->4] %[[RESHAPE_IN]] out_sharding=<@mesh, [{"z"}, {}, {"y"}, {}, {"x"}, {}]> : tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 2->5] %[[A2A1]] out_sharding=<@mesh, [{"z"}, {}, {}, {}, {"x"}, {"y"}]> : tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {"x"}, {"y"}]>]>} : (tensor<2x2x2x2x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @two_axis_scatter_with_untouched_axis(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z", "x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"x"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"z", "y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh, [{"z", "y"}, {"x"}, {}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh, [{"z"}, {"x"}, {"y"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// Tests full scatter on a non-major split dimension (dimension 1) where axes
// "x" and "y" are permuted and communicated to target dimensions before and
// after the split dimension (dimensions 0 and 2).
// CHECK-LABEL: func @non_major_split_dim
// CHECK-SAME:    %arg0: tensor<8x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x", "y"}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"x"}, {"y"}, {}, {}]>]>} : (tensor<8x16x8xf32>) -> tensor<8x2x2x4x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 1->0] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{"x"}, {}, {"y"}, {}, {}]> : tensor<8x2x2x4x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 2->4] %[[A2A1]] out_sharding=<@mesh_2d, [{"x"}, {}, {}, {}, {"y"}]> : tensor<8x2x2x4x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {}, {"y"}]>]>} : (tensor<8x2x2x4x8xf32>) -> tensor<8x16x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<8x16x8xf32>
func.func @non_major_split_dim(%arg0: tensor<8x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x", "y"}, {}]>}) -> (tensor<8x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x"}, {}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{}, {"y", "x"}, {}]> : tensor<8x16x8xf32>
  %1 = sdy.all_to_all [{"x"}: 1->0] %0 out_sharding=<@mesh_2d, [{"x"}, {"y"}, {}]> : tensor<8x16x8xf32>
  %2 = sdy.all_to_all [{"y"}: 1->2] %1 out_sharding=<@mesh_2d, [{"x"}, {}, {"y"}]> : tensor<8x16x8xf32>
  return %2 : tensor<8x16x8xf32>
}

// Tests sub-axis decomposition where sub-axis "x":(1)2 and axis "y" on
// split dimension 0 are permuted and fully scattered to other dimensions.
// CHECK-LABEL: func @sub_axis_full_scatter
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_sub, [{"x":(1)2, "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_sub, [{"x":(1)2}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x":(1)2}: 0->4] %[[RESHAPE_IN]] out_sharding=<@mesh_sub, [{}, {"y"}, {}, {}, {"x":(1)2}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->3] %[[A2A1]] out_sharding=<@mesh_sub, [{}, {}, {}, {"y"}, {"x":(1)2}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_sub, [{}, {"y"}, {"x":(1)2}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @sub_axis_full_scatter(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_sub, [{"x":(1)2, "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_sub, [{}, {"y"}, {"x":(1)2}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_sub, [{"y", "x":(1)2}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x":(1)2}: 0->2] %0 out_sharding=<@mesh_sub, [{"y"}, {}, {"x":(1)2}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_sub, [{}, {"y"}, {"x":(1)2}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// Tests that the all-to-all chain extraction stops at split dimension
// boundaries, rewriting the chain on dim 0 while leaving the downstream
// AllToAllOp sourcing from dim 1 intact.
// CHECK-LABEL: func @downstream_all_to_all_on_other_dim
// CHECK-SAME:    %arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {"z"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}, {"z"}, {}, {}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<2x2x4x8x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->4] %[[RESHAPE_IN]] out_sharding=<@mesh, [{}, {"y"}, {}, {"z"}, {"x"}, {}]> : tensor<2x2x4x8x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->5] %[[A2A1]] out_sharding=<@mesh, [{}, {}, {}, {"z"}, {"x"}, {"y"}]> : tensor<2x2x4x8x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}, {"x"}, {"y"}]>]>} : (tensor<2x2x4x8x8x8xf32>) -> tensor<16x8x8x8xf32>
// CHECK:       %[[DOWNSTREAM_A2A:.*]] = sdy.all_to_all [{"z"}: 1->2] %[[RESHAPE_OUT]] out_sharding=<@mesh, [{}, {}, {"x", "z"}, {"y"}]> : tensor<16x8x8x8xf32>
// CHECK:       return %[[DOWNSTREAM_A2A]] : tensor<16x8x8x8xf32>
func.func @downstream_all_to_all_on_other_dim(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {"z"}, {}, {}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"x", "z"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"y", "x"}, {"z"}, {}, {}]> : tensor<16x8x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh, [{"y"}, {"z"}, {"x"}, {}]> : tensor<16x8x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->3] %1 out_sharding=<@mesh, [{}, {"z"}, {"x"}, {"y"}]> : tensor<16x8x8x8xf32>
  %3 = sdy.all_to_all [{"z"}: 1->2] %2 out_sharding=<@mesh, [{}, {}, {"x", "z"}, {"y"}]> : tensor<16x8x8x8xf32>
  return %3 : tensor<16x8x8x8xf32>
}

