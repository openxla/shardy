// RUN: sdy_opt %s 2>&1 | FileCheck %s

sdy.mesh @mesh1 = <["x"=2, "y"=2]>
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

// CHECK-LABEL: func @all_to_all_target_dim_empty
func.func @all_to_all_target_dim_empty(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all {"x"} 0->1 %arg0 out_sharding=<@mesh1, [{}, {"x"}]>
  %0 = sdy.all_to_all {"x"} 0->1 %arg0 out_sharding=<@mesh1, [{}, {"x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_target_dim_not_empty
func.func @all_to_all_target_dim_not_empty(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all {"y"} 1->0 %arg0 out_sharding=<@mesh1, [{"x", "y"}, {}]>
  %0 = sdy.all_to_all {"y"} 1->0 %arg0 out_sharding=<@mesh1, [{"x", "y"}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_multiple_axes
func.func @all_to_all_multiple_axes(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"z", "y", "x"}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all {"y", "x"} 0->1 %arg0 out_sharding=<@mesh2, [{"z"}, {"y", "x"}]>
  %0 = sdy.all_to_all {"y", "x"} 0->1 %arg0 out_sharding=<@mesh2, [{"z"}, {"y", "x"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_not_all_dims_involved
func.func @all_to_all_not_all_dims_involved(%arg0 : tensor<16x8x4x4xf32> {sdy.sharding=#sdy.sharding<@mesh5, [{"x"}, {"y", "z"}, {"w"}, {}]>}) -> tensor<16x8x4x4xf32> {
  // CHECK-NEXT: dy.all_to_all {"w"} 2->0 %arg0 out_sharding=<@mesh5, [{"x", "w"}, {"y", "z"}, {}, {}]>
  %0 = sdy.all_to_all {"w"} 2->0 %arg0 out_sharding=<@mesh5, [{"x", "w"}, {"y", "z"}, {}, {}]> : tensor<16x8x4x4xf32>
  return %0 : tensor<16x8x4x4xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_exact_match
func.func @all_to_all_subaxis_exact_match(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all {"x":(1)2} 1->0 %arg0 out_sharding=<@mesh3, [{"y", "x":(1)2}, {}]>
  %0 = sdy.all_to_all {"x":(1)2} 1->0 %arg0 out_sharding=<@mesh3, [{"y", "x":(1)2}, {}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_ignored
func.func @all_to_all_subaxis_ignored(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh3, [{"y"}, {"x":(1)2}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all {"y"} 0->1 %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2, "y"}]>
  %0 = sdy.all_to_all {"y"} 0->1 %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2, "y"}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_suffix_of_full
func.func @all_to_all_subaxis_suffix_of_full(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"x", "z"}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all {"x":(4)2, "z"} 1->0 %arg0 out_sharding=<@mesh4, [{"y", "x":(4)2, "z"}, {"x":(1)4}]>
  %0 = sdy.all_to_all {"x":(4)2, "z"} 1->0 %arg0 out_sharding=<@mesh4, [{"y", "x":(4)2, "z"}, {"x":(1)4}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_to_all_subaxis_suffix_of_subaxis
func.func @all_to_all_subaxis_suffix_of_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh4, [{"y"}, {"z", "x":(1)4}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.all_to_all {"x":(2)2} 1->0 %arg0 out_sharding=<@mesh4, [{"y", "x":(2)2}, {"z", "x":(1)2}]>
  %0 = sdy.all_to_all {"x":(2)2} 1->0 %arg0 out_sharding=<@mesh4, [{"y", "x":(2)2}, {"z", "x":(1)2}]> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
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

// CHECK-LABEL: func @all_reduce
func.func @all_reduce(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh1, [{}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh1, [{}, {"x"}]> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh1, [{}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

sdy.mesh @mesh_xyzw = <["x"=2, "y"=2, "z"=2, "w"=2]>

// CHECK-LABEL: func @all_reduce_many_axes
func.func @all_reduce_many_axes(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh_xyzw, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {"z", "w"} %arg0 out_sharding=<@mesh_xyzw, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {"z", "w"} %arg0 out_sharding=<@mesh_xyzw, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_reduce_split_axis
func.func @all_reduce_split_axis(%arg0 : tensor<16x32xf32> {sdy.sharding=#sdy.sharding<@mesh7, [{"y"}, {"x": (2)4}]>}) -> tensor<16x32xf32> {
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

// CHECK-LABEL: func @all_reduce_output_is_explicitely_replicated
func.func @all_reduce_output_is_explicitely_replicated(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {"x", "y"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: sdy.all_reduce {} %arg0 out_sharding=<@mesh2, [{}, {"x", "y"}], replicated={"z"}> :  tensor<16x2xf32>
  %0 = sdy.all_reduce {} %arg0 out_sharding=<@mesh2, [{}, {"x", "y"}], replicated={"z"}> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}
