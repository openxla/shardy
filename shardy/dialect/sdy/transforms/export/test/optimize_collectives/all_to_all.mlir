// RUN: sdy_opt %s -sdy-optimize-collectives | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_2d = <["x"=2, "y"=2]>
sdy.mesh @mesh_3d = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_prime = <["x"=3, "y"=5, "z"=2]>
sdy.mesh @mesh_sub = <["x"=4, "y"=2]>
sdy.mesh @mesh_subaxis = <["x"=8]>
sdy.mesh @mesh_subaxis_16 = <["x"=16]>
sdy.mesh @mesh_4d = <["a"=2, "b"=2, "c"=2, "d"=2]>
sdy.mesh @mesh_4d_xyzw = <["x"=2, "y"=2, "z"=2, "w"=2]>
sdy.mesh @mesh_with_target = <["x"=2, "y"=2, "w"=2]>
sdy.mesh @mesh_custom_devs = <["x"=2, "y"=2], device_ids=[0, 2, 1, 3]>

// =============================================================================
// Category 1: Multi-Axis Permutations & Full Scatters
// =============================================================================

// CHECK-LABEL: func @two_axis_full_scatter
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->4, {"y"}: 1->3] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {}, {}, {"y"}, {"x"}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"y"}, {"x"}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @two_axis_full_scatter(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @three_axis_full_scatter
// CHECK-SAME:    %arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y", "z"}, {}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {"z"}, {}, {}, {}, {}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->6, {"y"}: 1->5, {"z"}: 2->4] %[[RESHAPE_IN]] out_sharding=<@mesh, [{}, {}, {}, {}, {"z"}, {"y"}, {"x"}]> : tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}, {"y"}, {"x"}]>]>} : (tensor<2x2x2x2x8x8x8xf32>) -> tensor<16x8x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8x8xf32>
func.func @three_axis_full_scatter(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y", "z"}, {}, {}, {}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"z"}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"z", "y", "x"}, {}, {}, {}]> : tensor<16x8x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->3] %0 out_sharding=<@mesh, [{"z", "y"}, {}, {}, {"x"}]> : tensor<16x8x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh, [{"z"}, {}, {"y"}, {"x"}]> : tensor<16x8x8x8xf32>
  %3 = sdy.all_to_all [{"z"}: 0->1] %2 out_sharding=<@mesh, [{}, {"z"}, {"y"}, {"x"}]> : tensor<16x8x8x8xf32>
  return %3 : tensor<16x8x8x8xf32>
}

// // Cyclic permutation {"x", "y", "z"} -> {"z", "x", "y"} where ALL 3 are communicated.
// CHECK-LABEL: func @safe_three_axis_full_scatter
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"x"}, {"y"}, {"z"}, {}, {}, {}, {}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->4, {"y"}: 1->5, {"z"}: 2->6] %[[RESHAPE_IN]] out_sharding=<@mesh_3d, [{}, {}, {}, {}, {"x"}, {"y"}, {"z"}]> : tensor<2x2x2x2x8x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{}, {"x"}, {"y"}, {"z"}]>]>} : (tensor<2x2x2x2x8x8x8xf32>) -> tensor<16x8x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8x8xf32>
func.func @safe_three_axis_full_scatter(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y", "z"}, {}, {}, {}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{}, {"x"}, {"y"}, {"z"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"z", "x", "y"}, {}, {}, {}]> : tensor<16x8x8x8xf32>
  %1 = sdy.all_to_all [{"y"}: 0->2] %0 out_sharding=<@mesh_3d, [{"z", "x"}, {}, {"y"}, {}]> : tensor<16x8x8x8xf32>
  %2 = sdy.all_to_all [{"x"}: 0->1] %1 out_sharding=<@mesh_3d, [{"z"}, {"x"}, {"y"}, {}]> : tensor<16x8x8x8xf32>
  %3 = sdy.all_to_all [{"z"}: 0->3] %2 out_sharding=<@mesh_3d, [{}, {"x"}, {"y"}, {"z"}]> : tensor<16x8x8x8xf32>
  return %3 : tensor<16x8x8x8xf32>
}

// CHECK-LABEL: func @identity_permutation
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->3, {"y"}: 1->4] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {}, {}, {"x"}, {"y"}]> : tensor<2x2x4x8x8xf32>
// CHECK-NOT:   stablehlo.transpose
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"x"}, {"y"}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @identity_permutation(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"x", "y"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"y"}: 0->2] %0 out_sharding=<@mesh_2d, [{"x"}, {}, {"y"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"x"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"x"}, {"y"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @non_zero_split_dim
// CHECK-SAME:    %arg0: tensor<8x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x", "y"}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"x"}, {"y"}, {}, {}]>]>} : (tensor<8x16x8xf32>) -> tensor<8x2x2x4x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 1->0, {"y"}: 2->4] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{"x"}, {}, {}, {}, {"y"}]> : tensor<8x2x2x4x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {}, {"y"}]>]>} : (tensor<8x2x2x4x8xf32>) -> tensor<8x16x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<8x16x8xf32>
func.func @non_zero_split_dim(%arg0: tensor<8x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"x", "y"}, {}]>}) -> (tensor<8x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x"}, {}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{}, {"y", "x"}, {}]> : tensor<8x16x8xf32>
  %1 = sdy.all_to_all [{"x"}: 1->0] %0 out_sharding=<@mesh_2d, [{"x"}, {"y"}, {}]> : tensor<8x16x8xf32>
  %2 = sdy.all_to_all [{"y"}: 1->2] %1 out_sharding=<@mesh_2d, [{"x"}, {}, {"y"}]> : tensor<8x16x8xf32>
  return %2 : tensor<8x16x8xf32>
}

// CHECK-LABEL: func @exact_divisibility_no_residual
// CHECK-SAME:    %arg0: tensor<4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<4x8x8xf32>) -> tensor<2x2x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->3, {"y"}: 1->2] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {}, {"y"}, {"x"}]> : tensor<2x2x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"y"}, {"x"}]>]>} : (tensor<2x2x8x8xf32>) -> tensor<4x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<4x8x8xf32>
func.func @exact_divisibility_no_residual(%arg0: tensor<4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<4x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<4x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<4x8x8xf32>
  return %2 : tensor<4x8x8xf32>
}

// =============================================================================
// Category 2: Sub-Axis Decomposition Edge Cases
// =============================================================================

// CHECK-LABEL: func @sub_axes_decomposition
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_sub, [{"x":(1)2, "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_sub, [{"x":(1)2}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x":(1)2}: 0->4, {"y"}: 1->3] %[[RESHAPE_IN]] out_sharding=<@mesh_sub, [{}, {}, {}, {"y"}, {"x":(1)2}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_sub, [{}, {"y"}, {"x":(1)2}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @sub_axes_decomposition(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_sub, [{"x":(1)2, "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_sub, [{}, {"y"}, {"x":(1)2}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_sub, [{"y", "x":(1)2}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x":(1)2}: 0->2] %0 out_sharding=<@mesh_sub, [{"y"}, {}, {"x":(1)2}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_sub, [{}, {"y"}, {"x":(1)2}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// Sub-axes "x":(1)2 and "x":(4)2 permuted and both communicated.
// CHECK-LABEL: func @sub_axis_decomposition_full_scatter
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis_16, [{"x":(1)2}, {"x":(4)2}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x":(1)2}: 0->3, {"x":(4)2}: 1->4] %[[RESHAPE_IN]] out_sharding=<@mesh_subaxis_16, [{}, {}, {}, {"x":(1)2}, {"x":(4)2}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis_16, [{}, {"x":(1)2}, {"x":(4)2}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @sub_axis_decomposition_full_scatter(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(1)2, "x":(4)2}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{}, {"x":(1)2}, {"x":(4)2}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_subaxis_16, [{"x":(4)2, "x":(1)2}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x":(1)2}: 0->1] %0 out_sharding=<@mesh_subaxis_16, [{"x":(4)2}, {"x":(1)2}, {}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"x":(4)2}: 0->2] %1 out_sharding=<@mesh_subaxis_16, [{}, {"x":(1)2}, {"x":(4)2}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// GCD sub-axis decomposition where collective permute decomposes "x":(1)4 into {"x":(1)2, "x":(2)2}.
// CHECK-LABEL: func @gcd_sub_axis_decomposition
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis, [{"x":(1)2}, {"x":(2)2}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x":(1)2}: 0->3, {"x":(2)2}: 1->4] %[[RESHAPE_IN]] out_sharding=<@mesh_subaxis, [{}, {}, {}, {"x":(1)2}, {"x":(2)2}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_subaxis, [{}, {"x":(1)2}, {"x":(2)2}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @gcd_sub_axis_decomposition(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis, [{"x":(1)4}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis, [{}, {"x":(1)2}, {"x":(2)2}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_subaxis, [{"x":(2)2, "x":(1)2}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x":(1)2}: 0->1] %0 out_sharding=<@mesh_subaxis, [{"x":(2)2}, {"x":(1)2}, {}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"x":(2)2}: 0->2] %1 out_sharding=<@mesh_subaxis, [{}, {"x":(1)2}, {"x":(2)2}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// =============================================================================
// Category 3: Layout, Local Transposition & Pre-Existing Sharding
// =============================================================================

// "x" and "y" permuted and communicated; unpermuted "z" stays on Dim 0.
// CHECK-LABEL: func @safe_partial_permute_unpermuted_stays
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"z"}, {"x"}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 1->4, {"y"}: 2->5] %[[RESHAPE_IN]] out_sharding=<@mesh_3d, [{"z"}, {}, {}, {}, {"x"}, {"y"}]> : tensor<2x2x2x2x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3d, [{"z"}, {"x"}, {"y"}]>]>} : (tensor<2x2x2x2x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
func.func @safe_partial_permute_unpermuted_stays(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"z", "x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"z"}, {"x"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"z", "y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_3d, [{"z", "y"}, {"x"}, {}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh_3d, [{"z"}, {"x"}, {"y"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// Non-power-of-two mesh axes (3x5x2 = 30) on Dim 0 of size 60 (Remainder size 2).
// CHECK-LABEL: func @asymmetric_prime_mesh_with_remainder
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_prime, [{"z"}, {"x"}, {"y"}, {}, {}, {}]>]>} : (tensor<60x10x10xf32>) -> tensor<2x3x5x2x10x10xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 1->4, {"y"}: 2->5] %[[RESHAPE_IN]] out_sharding=<@mesh_prime, [{"z"}, {}, {}, {}, {"x"}, {"y"}]> : tensor<2x3x5x2x10x10xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_prime, [{"z"}, {"x"}, {"y"}]>]>} : (tensor<2x3x5x2x10x10xf32>) -> tensor<60x10x10xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<60x10x10xf32>
func.func @asymmetric_prime_mesh_with_remainder(%arg0: tensor<60x10x10xf32> {sdy.sharding = #sdy.sharding<@mesh_prime, [{"z", "x", "y"}, {}, {}]>}) -> (tensor<60x10x10xf32> {sdy.sharding = #sdy.sharding<@mesh_prime, [{"z"}, {"x"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_prime, [{"z", "y", "x"}, {}, {}]> : tensor<60x10x10xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_prime, [{"z", "y"}, {"x"}, {}]> : tensor<60x10x10xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh_prime, [{"z"}, {"x"}, {"y"}]> : tensor<60x10x10xf32>
  return %2 : tensor<60x10x10xf32>
}

// 4-Axis Mesh. "a" and "b" leave Dim 0, leaving "c" and "d" behind.
// CHECK-LABEL: func @multi_axis_departed_stride_inversion
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4d, [{"c"}, {"d"}, {"a"}, {"b"}, {}, {}, {}]>]>} : (tensor<32x8x8xf32>) -> tensor<2x2x2x2x2x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"a"}: 2->5, {"b"}: 3->6] %[[RESHAPE_IN]] out_sharding=<@mesh_4d, [{"c"}, {"d"}, {}, {}, {}, {"a"}, {"b"}]> : tensor<2x2x2x2x2x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4d, [{"c", "d"}, {"a"}, {"b"}]>]>} : (tensor<2x2x2x2x2x8x8xf32>) -> tensor<32x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<32x8x8xf32>
func.func @multi_axis_departed_stride_inversion(%arg0: tensor<32x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4d, [{"c", "d", "a", "b"}, {}, {}]>}) -> (tensor<32x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4d, [{"c", "d"}, {"a"}, {"b"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_4d, [{"c", "d", "b", "a"}, {}, {}]> : tensor<32x8x8xf32>
  %1 = sdy.all_to_all [{"a"}: 0->1] %0 out_sharding=<@mesh_4d, [{"c", "d", "b"}, {"a"}, {}]> : tensor<32x8x8xf32>
  %2 = sdy.all_to_all [{"b"}: 0->2] %1 out_sharding=<@mesh_4d, [{"c", "d"}, {"a"}, {"b"}]> : tensor<32x8x8xf32>
  return %2 : tensor<32x8x8xf32>
}

// Target dimension is already sharded by axis "w".
// CHECK-LABEL: func @target_dimension_pre_sharded
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_with_target, [{"x"}, {"y"}, {}, {"w"}, {}]>]>} : (tensor<16x16x16xf32>) -> tensor<2x2x4x16x16xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->3, {"y"}: 1->4] %[[RESHAPE_IN]] out_sharding=<@mesh_with_target, [{}, {}, {}, {"w", "x"}, {"y"}]> : tensor<2x2x4x16x16xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_with_target, [{}, {"w", "x"}, {"y"}]>]>} : (tensor<2x2x4x16x16xf32>) -> tensor<16x16x16xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x16x16xf32>
func.func @target_dimension_pre_sharded(%arg0: tensor<16x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_with_target, [{"x", "y"}, {"w"}, {}]>}) -> (tensor<16x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_with_target, [{}, {"w", "x"}, {"y"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_with_target, [{"y", "x"}, {"w"}, {}]> : tensor<16x16x16xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_with_target, [{"y"}, {"w", "x"}, {}]> : tensor<16x16x16xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh_with_target, [{}, {"w", "x"}, {"y"}]> : tensor<16x16x16xf32>
  return %2 : tensor<16x16x16xf32>
}

// CHECK-LABEL: func @unaffected_shardings_on_non_split_dims
// CHECK-SAME:    %arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {"z"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}, {"z"}, {}, {}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<2x2x4x8x8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->5, {"y"}: 1->4] %[[RESHAPE_IN]] out_sharding=<@mesh, [{}, {}, {}, {"z"}, {"y"}, {"x"}]> : tensor<2x2x4x8x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}, {"y"}, {"x"}]>]>} : (tensor<2x2x4x8x8x8xf32>) -> tensor<16x8x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8x8xf32>
func.func @unaffected_shardings_on_non_split_dims(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {"z"}, {}, {}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"z"}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"y", "x"}, {"z"}, {}, {}]> : tensor<16x8x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->3] %0 out_sharding=<@mesh, [{"y"}, {"z"}, {}, {"x"}]> : tensor<16x8x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->2] %1 out_sharding=<@mesh, [{}, {"z"}, {"y"}, {"x"}]> : tensor<16x8x8x8xf32>
  return %2 : tensor<16x8x8x8xf32>
}

// =============================================================================
// Category 4: Negative Bailout Cases (Optimization must be safely skipped)
// =============================================================================

// Permuted axis "y" is NOT communicated -> Must keep CollectivePermute.
// CHECK-LABEL: func @unsafe_three_axis_partial_comm
// CHECK:       sdy.collective_permute
// CHECK:       sdy.all_to_all [{"z"}: 0->2]
// CHECK:       sdy.all_to_all [{"x"}: 0->1]
func.func @unsafe_three_axis_partial_comm(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y", "z"}, {}, {}]>}) -> tensor<16x8x8xf32> {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"y", "x", "z"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"z"}: 0->2] %0 out_sharding=<@mesh_3d, [{"y", "x"}, {}, {"z"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"x"}: 0->1] %1 out_sharding=<@mesh_3d, [{"y"}, {"x"}, {"z"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @bailout_permuted_subset_violation
// CHECK:       %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}]> : tensor<8x8xf32>
// CHECK:       %[[A2A:.*]] = sdy.all_to_all [{"x"}: 0->1] %[[CP]] out_sharding=<@mesh_2d, [{"y"}, {"x"}]> : tensor<8x8xf32>
// CHECK:       return %[[A2A]] : tensor<8x8xf32>
func.func @bailout_permuted_subset_violation(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}]> : tensor<8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_2d, [{"y"}, {"x"}]> : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// Sub-axis "x":(4)2 permuted but NOT communicated -> Must keep CollectivePermute.
// CHECK-LABEL: func @sub_axis_partial_comm_scramble_guard
// CHECK:       sdy.collective_permute
// CHECK:       sdy.all_to_all
func.func @sub_axis_partial_comm_scramble_guard(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_subaxis_16, [{"x":(1)2, "x":(4)2}, {}]>}) -> tensor<16x8xf32> {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_subaxis_16, [{"x":(4)2, "x":(1)2}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"x":(1)2}: 0->1] %0 out_sharding=<@mesh_subaxis_16, [{"x":(4)2}, {"x":(1)2}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// Cyclic 3-axis permutation where only 1 axis is communicated.
// CHECK-LABEL: func @bailout_cyclic_partial_comm
// CHECK:       sdy.collective_permute
// CHECK:       sdy.all_to_all
func.func @bailout_cyclic_partial_comm(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_3d, [{"x", "y", "z"}, {}, {}, {}]>}) -> tensor<16x8x8x8xf32> {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_3d, [{"z", "x", "y"}, {}, {}, {}]> : tensor<16x8x8x8xf32>
  %1 = sdy.all_to_all [{"y"}: 0->2] %0 out_sharding=<@mesh_3d, [{"z", "x"}, {}, {"y"}, {}]> : tensor<16x8x8x8xf32>
  return %1 : tensor<16x8x8x8xf32>
}

// CHECK-LABEL: func @bailout_non_split_dim_modified
// CHECK:       %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_4d_xyzw, [{"y", "x"}, {"w", "z"}, {}]> : tensor<8x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->2] %[[CP]] out_sharding=<@mesh_4d_xyzw, [{"y"}, {"w", "z"}, {"x"}]> : tensor<8x8x8xf32>
// CHECK:       return %[[A2A1]] : tensor<8x8x8xf32>
func.func @bailout_non_split_dim_modified(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4d_xyzw, [{"x", "y"}, {"z", "w"}, {}]>}) -> (tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4d_xyzw, [{"y"}, {"w", "z"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_4d_xyzw, [{"y", "x"}, {"w", "z"}, {}]> : tensor<8x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_4d_xyzw, [{"y"}, {"w", "z"}, {"x"}]> : tensor<8x8x8xf32>
  return %1 : tensor<8x8x8xf32>
}

// Multiple axes targeting the same destination dimension.
// CHECK-LABEL: func @reject_duplicate_target_dimension
// CHECK:       sdy.collective_permute
// CHECK:       sdy.all_to_all [{"x"}: 0->1]
// CHECK:       sdy.all_to_all [{"y"}: 0->1]
func.func @reject_duplicate_target_dimension(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}]>}) -> tensor<16x16xf32> {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}]> : tensor<16x16xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_2d, [{"y"}, {"x"}]> : tensor<16x16xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"x", "y"}]> : tensor<16x16xf32>
  return %2 : tensor<16x16xf32>
}

// Static Divisibility Failure (Dim 0 size 10 is not divisible by 2*2 = 4).
// CHECK-LABEL: func @reject_non_divisible_static_shape
// CHECK:       sdy.collective_permute
// CHECK:       sdy.all_to_all
func.func @reject_non_divisible_static_shape(%arg0: tensor<10x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<10x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<10x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<10x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<10x8x8xf32>
  return %2 : tensor<10x8x8xf32>
}

// Dynamic shape dimension.
// CHECK-LABEL: func @bailout_dynamic_shape
// CHECK:       sdy.collective_permute
// CHECK:       sdy.all_to_all
func.func @bailout_dynamic_shape(%arg0: tensor<?x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> tensor<?x8x8xf32> {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<?x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<?x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<?x8x8xf32>
  return %2 : tensor<?x8x8xf32>
}

// Rank 0 (scalar) tensor.
// CHECK-LABEL: func @bailout_rank_0
// CHECK:       sdy.collective_permute
// CHECK:       return
func.func @bailout_rank_0(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_2d, []>}) -> tensor<f32> {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, []> : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @bailout_multiple_uses
// CHECK:       %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->2] %[[CP]] out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 0->1] %[[A2A1]] out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
// CHECK:       return %[[A2A2]], %[[CP]] : tensor<16x8x8xf32>, tensor<16x8x8xf32>
func.func @bailout_multiple_uses(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32>, tensor<16x8x8xf32>) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
  return %2, %0 : tensor<16x8x8xf32>, tensor<16x8x8xf32>
}

// Intermediate AllToAllOp has multiple uses.
// CHECK-LABEL: func @bailout_intermediate_all_to_all_multiple_uses
// CHECK:       %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->2] %[[CP]] out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 0->1] %[[A2A1]] out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
// CHECK:       return %[[A2A2]], %[[A2A1]] : tensor<16x8x8xf32>, tensor<16x8x8xf32>
func.func @bailout_intermediate_all_to_all_multiple_uses(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32>, tensor<16x8x8xf32>) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
  return %2, %1 : tensor<16x8x8xf32>, tensor<16x8x8xf32>
}

// CHECK-LABEL: func @bailout_mesh_device_id_changes
// CHECK:       %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_custom_devs, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->2] %[[CP]] out_sharding=<@mesh_custom_devs, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 0->1] %[[A2A1]] out_sharding=<@mesh_custom_devs, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
// CHECK:       return %[[A2A2]] : tensor<16x8x8xf32>
func.func @bailout_mesh_device_id_changes(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_custom_devs, [{}, {"y"}, {"x"}]>}) {
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_custom_devs, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_custom_devs, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_custom_devs, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}

