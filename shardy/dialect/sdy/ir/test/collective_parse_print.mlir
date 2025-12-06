// RUN: sdy_opt %s 2>&1 | FileCheck %s

sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh1_non_iota = <["x"=2, "y"=2], device_ids=[3, 2, 1, 0]>
sdy.mesh @mesh2 = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh3 = <["x"=4, "y"=2]>
sdy.mesh @mesh4 = <["x"=8, "y"=2, "z"=2]>
sdy.mesh @mesh5 = <["x"=2, "y"=2, "z"=4, "w"=4]>
sdy.mesh @mesh6 = <["x"=4, "y"=4]>
sdy.mesh @mesh7 = <["x"=16, "y"=2]>

// CHECK-LABEL: func @all_gather1
func.func @all_gather1(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {}]>
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_gather2
func.func @all_gather2(%arg0 : tensor<16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y", "x", "z"}]>}) -> tensor<16xf32> {
  // CHECK-NEXT: sdy.all_gather [{"x", "z"}] %arg0 out_sharding=<@mesh2, [{"y"}]>
  %0 = sdy.all_gather [{"x", "z"}] %arg0 out_sharding=<@mesh2, [{"y"}]> : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @all_gather3
func.func @all_gather3(%arg0 : tensor<16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y", "x", "z"}]>}) -> tensor<16xf32> {
  // CHECK-NEXT: sdy.all_gather [{"y", "x", "z"}] %arg0 out_sharding=<@mesh2, [{}]>
  %0 = sdy.all_gather [{"y", "x", "z"}] %arg0 out_sharding=<@mesh2, [{}]> : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @all_gather4
func.func @all_gather4(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y", "x"}, {"z"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{"x"}, {"z"}] %arg0 out_sharding=<@mesh2, [{"y"}, {}]>
  %0 = sdy.all_gather [{"x"}, {"z"}] %arg0 out_sharding=<@mesh2, [{"y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_gather_subaxis_exact_match
func.func @all_gather_subaxis_exact_match(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {}]>
  %0 = sdy.all_gather [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_gather_subaxis_ignored
func.func @all_gather_subaxis_ignored(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2}]>
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_gather_subaxis_suffix_of_full
func.func @all_gather_subaxis_suffix_of_full(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"x", "z"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x":(1)4}]>
  %0 = sdy.all_gather [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x":(1)4}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_gather_subaxis_suffix_of_subaxis
func.func @all_gather_subaxis_suffix_of_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"z", "x":(1)4}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_gather [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{"y"}, {"z", "x":(1)2}]>
  %0 = sdy.all_gather [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{"y"}, {"z", "x":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice1
func.func @all_slice1(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]>
  %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice2
func.func @all_slice2(%arg0 : tensor<16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y"}]>}) -> tensor<16xf32> {
  // CHECK-NEXT: sdy.all_slice [{"x", "z"}] %arg0 out_sharding=<@mesh2, [{"y", "x", "z"}]>
  %0 = sdy.all_slice [{"x", "z"}] %arg0 out_sharding=<@mesh2, [{"y", "x", "z"}]> : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @all_slice3
func.func @all_slice3(%arg0 : tensor<16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}]>}) -> tensor<16xf32> {
  // CHECK-NEXT: sdy.all_slice [{"y", "x", "z"}] %arg0 out_sharding=<@mesh2, [{"y", "x", "z"}]>
  %0 = sdy.all_slice [{"y", "x", "z"}] %arg0 out_sharding=<@mesh2, [{"y", "x", "z"}]> : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @all_slice4
func.func @all_slice4(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{"x"}, {"z"}] %arg0 out_sharding=<@mesh2, [{"y", "x"}, {"z"}]>
  %0 = sdy.all_slice [{"x"}, {"z"}] %arg0 out_sharding=<@mesh2, [{"y", "x"}, {"z"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice_missing_in_sharding
func.func @all_slice_missing_in_sharding(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{}, {"x"}]>
  %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice_subaxis_exact_match
func.func @all_slice_subaxis_exact_match(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {"x":(1)2}]>
  %0 = sdy.all_slice [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {"x":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice_subaxis_ignored
func.func @all_slice_subaxis_ignored(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{"y"}, {}] %arg0 out_sharding=<@mesh3, [{"y"}, {"x":(1)2}]>
  %0 = sdy.all_slice [{"y"}, {}] %arg0 out_sharding=<@mesh3, [{"y"}, {"x":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice_subaxis_suffix_of_full
func.func @all_slice_subaxis_suffix_of_full(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"x":(1)4}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x", "z"}]>
  %0 = sdy.all_slice [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x", "z"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice_subaxis_suffix_of_subaxis
func.func @all_slice_subaxis_suffix_of_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"z", "x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_slice [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{"y"}, {"z", "x":(1)4}]>
  %0 = sdy.all_slice [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{"y"}, {"z", "x":(1)4}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_target_dim_empty_single_param
func.func @all_to_all_target_dim_empty_single_param(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"x"}: 0->1] %arg0 out_sharding=<@mesh1, [{}, {"x"}]>
  %0 = sdy.all_to_all [{"x"}: 0->1] %arg0 out_sharding=<@mesh1, [{}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_target_dim_empty_multiple_params
func.func @all_to_all_target_dim_empty_multiple_params(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}, {"y"}, {}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"x"}: 0->2, {"y"}: 1->3] %arg0 out_sharding=<@mesh1, [{}, {}, {"x"}, {"y"}]>
  %0 = sdy.all_to_all [{"x"}: 0->2, {"y"}: 1->3] %arg0 out_sharding=<@mesh1, [{}, {}, {"x"}, {"y"}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_target_dim_not_empty_single_param
func.func @all_to_all_target_dim_not_empty_single_param(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"y"}: 1->0] %arg0 out_sharding=<@mesh1, [{"x", "y"}, {}]>
  %0 = sdy.all_to_all [{"y"}: 1->0] %arg0 out_sharding=<@mesh1, [{"x", "y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_target_dim_not_empty_multiple_params
func.func @all_to_all_target_dim_not_empty_multiple_params(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"x"}, {"y"}, {"z"}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"x"}: 0->1, {"z"}: 2->3] %arg0 out_sharding=<@mesh2, [{}, {"y", "x"}, {}, {"z"}]>
  %0 = sdy.all_to_all [{"x"}: 0->1, {"z"}: 2->3] %arg0 out_sharding=<@mesh2, [{}, {"y", "x"}, {}, {"z"}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_multiple_axes_single_param
func.func @all_to_all_multiple_axes_single_param(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"z", "y", "x"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"y", "x"}: 0->1] %arg0 out_sharding=<@mesh2, [{"z"}, {"y", "x"}]>
  %0 = sdy.all_to_all [{"y", "x"}: 0->1] %arg0 out_sharding=<@mesh2, [{"z"}, {"y", "x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_multiple_axes_multiple_params
func.func @all_to_all_multiple_axes_multiple_params(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y", "z"}, {}, {"x"}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"y", "z"}: 0->1, {"x"}: 2->3] %arg0 out_sharding=<@mesh2, [{}, {"y", "z"}, {}, {"x"}]>
  %0 = sdy.all_to_all [{"y", "z"}: 0->1, {"x"}: 2->3] %arg0 out_sharding=<@mesh2, [{}, {"y", "z"}, {}, {"x"}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_not_all_dims_involved_single_param
func.func @all_to_all_not_all_dims_involved_single_param(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh5, [{"x"}, {"y", "z"}, {"w"}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: dy.all_to_all [{"w"}: 2->0] %arg0 out_sharding=<@mesh5, [{"x", "w"}, {"y", "z"}, {}, {}]>
  %0 = sdy.all_to_all [{"w"}: 2->0] %arg0 out_sharding=<@mesh5, [{"x", "w"}, {"y", "z"}, {}, {}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_not_all_dims_involved_multiple_params
func.func @all_to_all_not_all_dims_involved_multiple_params(%arg0 : tensor<16x8x4x4x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh5, [{"x"}, {"y", "z"}, {"w"}, {}, {}, {}]>}) -> tensor<16x8x4x4x4x4xf32> {
  // CHECK-NEXT: dy.all_to_all [{"x"}: 0->2, {"z"}: 1->3] %arg0 out_sharding=<@mesh5, [{}, {"y"}, {"w", "x"}, {"z"}, {}, {}]>
  %0 = sdy.all_to_all [{"x"}: 0->2, {"z"}: 1->3] %arg0 out_sharding=<@mesh5, [{}, {"y"}, {"w", "x"}, {"z"}, {}, {}]> : tensor<16x8x4x4x4x4xf32>
  return %0 : tensor<16x8x4x4x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_exact_match_single_param
func.func @all_to_all_subaxis_exact_match_single_param(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"x":(1)2}: 1->0] %arg0 out_sharding=<@mesh3, [{"y", "x":(1)2}, {}]>
  %0 = sdy.all_to_all [{"x":(1)2}: 1->0] %arg0 out_sharding=<@mesh3, [{"y", "x":(1)2}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_exact_match_multiple_params
func.func @all_to_all_subaxis_exact_match_multiple_params(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}, {}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"y"}: 0->2, {"x":(1)2}: 1->3] %arg0 out_sharding=<@mesh3, [{}, {}, {"y"}, {"x":(1)2}]> : tensor<16x8x4x4xf32>
  %0 = sdy.all_to_all [{"y"}: 0->2, {"x":(1)2}: 1->3] %arg0 out_sharding=<@mesh3, [{}, {}, {"y"}, {"x":(1)2}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_ignored_single_param
func.func @all_to_all_subaxis_ignored_single_param(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"y"}: 0->1] %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2, "y"}]>
  %0 = sdy.all_to_all [{"y"}: 0->1] %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2, "y"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_ignored_multiple_params
func.func @all_to_all_subaxis_ignored_multiple_params(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}, {}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"y"}: 0->2, {"x":(1)2}: 1->3] %arg0 out_sharding=<@mesh3, [{}, {}, {"y"}, {"x":(1)2}]>
  %0 = sdy.all_to_all [{"y"}: 0->2, {"x":(1)2}: 1->3] %arg0 out_sharding=<@mesh3, [{}, {}, {"y"}, {"x":(1)2}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_suffix_of_full_single_param
func.func @all_to_all_subaxis_suffix_of_full_single_param(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"x", "z"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"x":(4)2, "z"}: 1->0] %arg0 out_sharding=<@mesh4, [{"y", "x":(4)2, "z"}, {"x":(1)4}]>
  %0 = sdy.all_to_all [{"x":(4)2, "z"}: 1->0] %arg0 out_sharding=<@mesh4, [{"y", "x":(4)2, "z"}, {"x":(1)4}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_suffix_of_full_multiple_params
func.func @all_to_all_subaxis_suffix_of_full_multiple_params(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh5, [{"z"}, {"w", "x"}, {}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"z":(2)2}: 0->2, {"w":(2)2, "x"}: 1->3] %arg0 out_sharding=<@mesh5, [{"z":(1)2}, {"w":(1)2}, {"z":(2)2}, {"w":(2)2, "x"}]>
  %0 = sdy.all_to_all [{"z":(2)2}: 0->2, {"w":(2)2, "x"}: 1->3] %arg0 out_sharding=<@mesh5, [{"z":(1)2}, {"w":(1)2}, {"z":(2)2}, {"w":(2)2, "x"}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_suffix_of_subaxis
func.func @all_to_all_subaxis_suffix_of_subaxis_single_param(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"z", "x":(1)4}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"x":(2)2}: 1->0] %arg0 out_sharding=<@mesh4, [{"y", "x":(2)2}, {"z", "x":(1)2}]>
  %0 = sdy.all_to_all [{"x":(2)2}: 1->0] %arg0 out_sharding=<@mesh4, [{"y", "x":(2)2}, {"z", "x":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_suffix_of_subaxis_multiple_params
func.func @all_to_all_subaxis_suffix_of_subaxis_multiple_params(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"z", "x":(1)4}, {}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: sdy.all_to_all [{"y"}: 0->2, {"x":(2)2}: 1->3] %arg0 out_sharding=<@mesh4, [{}, {"z", "x":(1)2}, {"y"}, {"x":(2)2}]>
  %0 = sdy.all_to_all [{"y"}: 0->2, {"x":(2)2}: 1->3] %arg0 out_sharding=<@mesh4, [{}, {"z", "x":(1)2}, {"y"}, {"x":(2)2}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @collective_permute_reorder_axes_single_dim
func.func @collective_permute_reorder_axes_single_dim(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"x", "y", "z"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.collective_permute %arg0 out_sharding=<@mesh2, [{"z", "x", "y", ?}, {}]>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh2, [{"z", "x", "y", ?}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @collective_permute_reorder_across_multiple_dim
func.func @collective_permute_reorder_across_multiple_dim(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"x", "y"}, {"z"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.collective_permute %arg0 out_sharding=<@mesh2, [{"z", "x"}, {"y"}]>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh2, [{"z", "x"}, {"y"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @collective_permute_swap_axes_between_dims
func.func @collective_permute_swap_axes_between_dims(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.collective_permute %arg0 out_sharding=<@mesh2, [{"y"}, {"x"}]>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh2, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @collective_permute_replace_axes_single_dim
func.func @collective_permute_replace_axes_single_dim(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh5, [{"x", "y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.collective_permute %arg0 out_sharding=<@mesh5, [{"z"}, {}]>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh5, [{"z"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @collective_permute_replace_sub_axes_multiple_dims
func.func @collective_permute_replace_sub_axes_multiple_dims(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh5, [{"x", "z"}, {"w"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.collective_permute %arg0 out_sharding=<@mesh5, [{"z":(1)2, "x", "y"}, {"z":(2)2, "w":(1)2}]>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh5, [{"z":(1)2, "x", "y"}, {"z":(2)2, "w":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @collective_permute_reorder_device_ids
func.func @collective_permute_reorder_device_ids(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x", "y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.collective_permute %arg0 out_sharding=<@mesh1_non_iota, [{"x", "y"}, {}]>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh1_non_iota, [{"x", "y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @collective_permute_reorder_device_ids_and_axes
func.func @collective_permute_reorder_device_ids_and_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1_non_iota, [{"x", "y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.collective_permute %arg0 out_sharding=<@mesh1, [{"y", "x"}, {}]>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh1, [{"y", "x"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_reduce
func.func @all_reduce(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh1, [{}, {"x"}]> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh1, [{}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_reduce_unreduced_in_sharding
func.func @all_reduce_unreduced_in_sharding(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {"x"}], unreduced={"y", "z"}>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh2, [{}, {"x"}], replicated={"y"}, unreduced={"z"}> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh2, [{}, {"x"}], replicated={"y"}, unreduced={"z"}> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_reduce_missing_in_sharding
func.func @all_reduce_missing_in_sharding(%arg0 : tensor<16x2xf32>) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh1, [{}, {}]> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh1, [{}, {}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

sdy.mesh @mesh_xyzw = <["x"=2, "y"=2, "z"=2, "w"=2]>

// CHECK-LABEL: func @all_reduce_many_axes
func.func @all_reduce_many_axes(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh_xyzw, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {"z", "w"} %arg0 out_sharding=<@mesh_xyzw, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {"z", "w"} %arg0 out_sharding=<@mesh_xyzw, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_reduce_sub_axis
func.func @all_reduce_sub_axis(%arg0 : tensor<16x32xf32> {sdy.sharding=#sdy.sharding<@mesh7, [{"y"}, {"x": (2)4}]>}) -> tensor<16x32xf32> {
  // CHECK-NEXT: sdy.all_reduce {"x":(1)2} %arg0 out_sharding=<@mesh7, [{"y"}, {"x":(2)4}]> :  tensor<16x32xf32>
  %0 = sdy.all_reduce {"x":(1)2} %arg0 out_sharding=<@mesh7, [{"y"}, {"x":(2)4}]> :  tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @all_reduce_split_axis_y
func.func @all_reduce_split_axis_y(%arg0 : tensor<16x32xf32> {sdy.sharding=#sdy.sharding<@mesh6, [{"y":(1)2}, {"x"}]>}) -> tensor<16x32xf32> {
  // CHECK-NEXT: sdy.all_reduce {"y":(2)2} %arg0 out_sharding=<@mesh6, [{"y":(1)2}, {"x"}]> :  tensor<16x32xf32>
  %0 = sdy.all_reduce {"y":(2)2} %arg0 out_sharding=<@mesh6, [{"y":(1)2}, {"x"}]> :  tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @all_reduce_output_is_explicitly_replicated
func.func @all_reduce_output_is_explicitly_replicated(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {"x", "y"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {} %arg0 out_sharding=<@mesh2, [{}, {"x", "y"}], replicated={"z"}> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {} %arg0 out_sharding=<@mesh2, [{}, {"x", "y"}], replicated={"z"}> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @reduce_scatter1
func.func @reduce_scatter1(%arg0 : tensor<16xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}]>}) -> tensor<16xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{"y"}] %arg0 out_sharding=<@mesh1, [{"x", "y"}]> : tensor<16xf32>
  %0 = sdy.reduce_scatter [{"y"}] %arg0 out_sharding=<@mesh1, [{"x", "y"}]> : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @reduce_scatter2
func.func @reduce_scatter2(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{}, {"y"}] %arg0 out_sharding=<@mesh1, [{"x"}, {"y"}]> : tensor<16x8xf32>
  %0 = sdy.reduce_scatter [{}, {"y"}] %arg0 out_sharding=<@mesh1, [{"x"}, {"y"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter3
func.func @reduce_scatter3(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{"z"}, {}] %arg0 out_sharding=<@mesh2, [{"x", "z"}, {"y"}]> : tensor<16x8xf32>
  %0 = sdy.reduce_scatter [{"z"}, {}] %arg0 out_sharding=<@mesh2, [{"x", "z"}, {"y"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_missing_in_sharding
func.func @reduce_scatter_missing_in_sharding(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{"y"}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> : tensor<16x8xf32>
  %0 = sdy.reduce_scatter [{"y"}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_reduce_scatter_axes
func.func @reduce_scatter_empty_reduce_scatter_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{}, {}] %arg0 out_sharding=<@mesh1, [{"x"}, {}]> : tensor<16x8xf32>
  %0 = sdy.reduce_scatter [{}, {}] %arg0 out_sharding=<@mesh1, [{"x"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_subaxis_exact_match
func.func @reduce_scatter_subaxis_exact_match(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {"x":(1)2}]> : tensor<16x8xf32>
  %0 = sdy.reduce_scatter [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {"x":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_subaxis_suffix_of_full
func.func @reduce_scatter_subaxis_suffix_of_full(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"x":(1)4}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x", "z"}]> : tensor<16x8xf32>
  %0 = sdy.reduce_scatter [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x", "z"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_subaxis_suffix_of_subaxis
func.func @reduce_scatter_subaxis_suffix_of_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"z", "x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reduce_scatter [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{"y"}, {"z", "x":(1)4}]> : tensor<16x8xf32>
  %0 = sdy.reduce_scatter [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{"y"}, {"z", "x":(1)4}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_1
func.func @sharded_to_unreduced_1(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"y"}, {"x"}]>}) -> tensor<16x8xf32> {
  %0 = sdy.sharded_to_unreduced [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {}], unreduced={"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_2
func.func @sharded_to_unreduced_2(%arg0 : tensor<16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y", "z", "x"}]>}) -> tensor<16xf32> {
  // CHECK-NEXT: sdy.sharded_to_unreduced [{"z", "x"}] %arg0 out_sharding=<@mesh2, [{"y"}], unreduced={"x", "z"}>
  %0 = sdy.sharded_to_unreduced [{"z", "x"}] %arg0 out_sharding=<@mesh2, [{"y"}], unreduced={"x", "z"}> : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_3
func.func @sharded_to_unreduced_3(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y", "z"}, {"x"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.sharded_to_unreduced [{"z"}, {"x"}] %arg0 out_sharding=<@mesh2, [{"y"}, {}], unreduced={"x", "z"}>
  %0 = sdy.sharded_to_unreduced [{"z"}, {"x"}] %arg0 out_sharding=<@mesh2, [{"y"}, {}], unreduced={"x", "z"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_input_has_unreduced_axes
func.func @sharded_to_unreduced_input_has_unreduced_axes(%arg0 : tensor<16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"z", "x"}], unreduced={"y"}>}) -> tensor<16xf32> {
  // CHECK-NEXT: sdy.sharded_to_unreduced [{"x"}] %arg0 out_sharding=<@mesh2, [{"z"}], unreduced={"x", "y"}>
  %0 = sdy.sharded_to_unreduced [{"x"}] %arg0 out_sharding=<@mesh2, [{"z"}], unreduced={"x", "y"}> : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_subaxis_exact_match
func.func @sharded_to_unreduced_subaxis_exact_match(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.sharded_to_unreduced [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {}], unreduced={"x":(1)2}>
  %0 = sdy.sharded_to_unreduced [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {}], unreduced={"x":(1)2}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_subaxis_suffix_of_full
func.func @sharded_to_unreduced_subaxis_suffix_of_full(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"x", "z"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.sharded_to_unreduced [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x":(1)4}], unreduced={"x":(4)2, "z"}>
  %0 = sdy.sharded_to_unreduced [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x":(1)4}], unreduced={"x":(4)2, "z"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_subaxis_suffix_of_subaxis
func.func @sharded_to_unreduced_subaxis_suffix_of_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"z", "x":(1)4}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.sharded_to_unreduced [{"y"}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{}, {"z", "x":(1)2}], unreduced={"x":(2)2, "y"}>
  %0 = sdy.sharded_to_unreduced [{"y"}, {"x":(2)2}] %arg0 out_sharding=<@mesh4, [{}, {"z", "x":(1)2}], unreduced={"x":(2)2, "y"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_sort_and_merge_axes
func.func @sharded_to_unreduced_sort_and_merge_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh6, [{"y":(1)2, "x":(2)2}, {"x":(1)2, "y":(2)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.sharded_to_unreduced [{"x":(2)2}, {"x":(1)2, "y":(2)2}] %arg0 out_sharding=<@mesh6, [{"y":(1)2}, {}], unreduced={"x", "y":(2)2}>
  %0 = sdy.sharded_to_unreduced [{"x":(2)2}, {"x":(1)2, "y":(2)2}] %arg0 out_sharding=<@mesh6, [{"y":(1)2}, {}], unreduced={"x", "y":(2)2}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
