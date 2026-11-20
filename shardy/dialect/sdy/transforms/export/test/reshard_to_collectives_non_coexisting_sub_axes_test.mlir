// RUN: sdy_opt %s -sdy-reshard-to-collectives | FileCheck %s

sdy.mesh @mesh1d_6 = <["x"=6]>
sdy.mesh @mesh1d_12 = <["x"=12]>
sdy.mesh @mesh2d_12x2 = <["x"=12, "y"=2]>

// CHECK-LABEL: func @non_coexisting_subaxes_same_pre_size
func.func @non_coexisting_subaxes_same_pre_size(%arg0 : tensor<6x2xf32> {sdy.sharding = #sdy.sharding<@mesh1d_6, [{"x":(1)2}, {}]>}) -> tensor<6x2xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(1)2}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{"x":(1)3}, {}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{"x":(1)3}, {}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{"x":(1)3}, {}]> : tensor<6x2xf32>
  return %0 : tensor<6x2xf32>
}

// CHECK-LABEL: func @non_coexisting_overlapping_subaxes_same_dim
func.func @non_coexisting_overlapping_subaxes_same_dim(%arg0 : tensor<6x2xf32> {sdy.sharding = #sdy.sharding<@mesh1d_6, [{"x":(2)3}, {}]>}) -> tensor<6x2xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(2)3}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{"x":(1)3}, {}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{"x":(1)3}, {}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{"x":(1)3}, {}]> : tensor<6x2xf32>
  return %0 : tensor<6x2xf32>
}

// CHECK-LABEL: func @non_coexisting_overlapping_subaxes_diff_dim
func.func @non_coexisting_overlapping_subaxes_diff_dim(%arg0 : tensor<6x2xf32> {sdy.sharding = #sdy.sharding<@mesh1d_6, [{"x":(2)3}, {}]>}) -> tensor<6x2xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(2)3}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"x":(1)3}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{}, {"x":(1)3}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{}, {"x":(1)3}]> : tensor<6x2xf32>
  return %0 : tensor<6x2xf32>
}

// CHECK-LABEL: func @disjoint_non_coexisting_subaxes_diff_dim
func.func @disjoint_non_coexisting_subaxes_diff_dim(%arg0 : tensor<6x6xf32> {sdy.sharding = #sdy.sharding<@mesh1d_6, [{"x":(1)2}, {}]>}) -> tensor<6x6xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(1)2}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"x":(3)2}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{}, {"x":(3)2}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{}, {"x":(3)2}]> : tensor<6x6xf32>
  return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func @disjoint_non_coexisting_subaxes_same_dim
func.func @disjoint_non_coexisting_subaxes_same_dim(%arg0 : tensor<6x6xf32> {sdy.sharding = #sdy.sharding<@mesh1d_6, [{"x":(1)2}, {}]>}) -> tensor<6x6xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(1)2}, {}] %arg0 out_sharding=<@mesh1d_6, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{"x":(3)2}, {}] %[[ALL_GATHER]] out_sharding=<@mesh1d_6, [{"x":(3)2}, {}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
  %0 = sdy.reshard %arg0 <@mesh1d_6, [{"x":(3)2}, {}]> : tensor<6x6xf32>
  return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func @unaligned_overlapping_subaxis_after_non_coexisting
func.func @unaligned_overlapping_subaxis_after_non_coexisting(%arg0 : tensor<12x12xf32> {sdy.sharding = #sdy.sharding<@mesh1d_12, [{"x":(1)3}, {"x":(6)2}]>}) -> tensor<12x12xf32> {
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(1)3}, {"x":(6)2}] %arg0 out_sharding=<@mesh1d_12, [{}, {}]>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"x":(2)6}] %[[ALL_GATHER]] out_sharding=<@mesh1d_12, [{}, {"x":(2)6}]>
  // CHECK-NEXT: return %[[ALL_SLICE]]
  %0 = sdy.reshard %arg0 <@mesh1d_12, [{}, {"x":(2)6}]> : tensor<12x12xf32>
  return %0 : tensor<12x12xf32>
}

// CHECK-LABEL: func @no_slice_for_all_to_all_on_dim_to_unshard
func.func @no_slice_for_all_to_all_on_dim_to_unshard(%arg0 : tensor<24x24xf32> {sdy.sharding = #sdy.sharding<@mesh2d_12x2, [{"x":(6)2, "x":(1)2}, {}]>}) -> tensor<24x24xf32> {
  // CHECK-NEXT: %[[SLICE_Y:.*]] = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh2d_12x2, [{"x":(6)2, "x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[ALL_GATHER:.*]] = sdy.all_gather [{"x":(6)2, "x":(1)2}, {}] %[[SLICE_Y]] out_sharding=<@mesh2d_12x2, [{}, {"y"}]>
  // CHECK-NEXT: %[[SLICE_X:.*]] = sdy.all_slice [{}, {"x":(1)4}] %[[ALL_GATHER]] out_sharding=<@mesh2d_12x2, [{}, {"y", "x":(1)4}]>
  // CHECK-NEXT: %[[CP:.*]] = sdy.collective_permute %[[SLICE_X]] out_sharding=<@mesh2d_12x2, [{}, {"x":(1)2, "y", "x":(2)2}]>
  // CHECK-NEXT: return %[[CP]]
  %0 = sdy.reshard %arg0 <@mesh2d_12x2, [{}, {"x":(1)2, "y", "x":(2)2}]> : tensor<24x24xf32>
  return %0 : tensor<24x24xf32>
}
