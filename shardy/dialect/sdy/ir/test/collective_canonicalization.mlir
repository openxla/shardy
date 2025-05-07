// RUN: sdy_opt %s -canonicalize | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=2, "y"=2], device_ids=[3, 2, 1, 0]>
sdy.mesh @mesh2 = <["x"=2, "y"=2, "z"=2, "p"=2, "q"=8, "r"=8]>

// CHECK-LABEL: func @null_all_gather
func.func @null_all_gather(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.all_gather [{}, {}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @null_all_slice
func.func @null_all_slice(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.all_slice [{}, {}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @null_all_reduce
func.func @null_all_reduce(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.all_reduce {} %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @null_collective_permute
func.func @null_collective_permute(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @collective_permute_reorder_device_ids
func.func @collective_permute_reorder_device_ids(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: %[[COLLECTIVE_PERMUTE:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_non_iota, [{"y"}, {"x"}]>
  // CHECK-NEXT: return %[[COLLECTIVE_PERMUTE]]
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_non_iota, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @null_collective_permute_with_diff_open_closed
func.func @null_collective_permute_with_diff_open_closed(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y", ?}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_slice_of_all_gather
func.func @all_slice_of_all_gather(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // TODO(kostiantynl): orphaned all_gather should be removed.
  // CHECK: %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  %1 = sdy.all_slice [{}, {"x"}] %0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %1 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_slice_of_all_gather_mismatching_axes_per_dim
func.func @all_slice_of_all_gather_mismatching_axes_per_dim(%arg0 : tensor<16x4xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x4xf32> {
  // CHECK: %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x"}]> :  tensor<16x4xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{}, {"y"}] %0 out_sharding=<@mesh, [{}, {"x", "y"}]> :  tensor<16x4xf32>
  // CHECK-NEXT: return %1 : tensor<16x4xf32>
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x"}]> :  tensor<16x4xf32>
  %1 = sdy.all_slice [{}, {"y"}] %0 out_sharding=<@mesh, [{}, {"x", "y"}]> :  tensor<16x4xf32>
  return %1 : tensor<16x4xf32>
}

// CHECK-LABEL: func @all_slice_of_all_gather_many_uses
func.func @all_slice_of_all_gather_many_uses(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  // CHECK: %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  // CHECK-NEXT: return %arg0, %0 : tensor<16x2xf32>, tensor<16x2xf32>
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  %1 = sdy.all_slice [{}, {"x"}] %0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %arg0, %0 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion
func.func @reduce_scatter_fusion(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  // CHECK-NEXT: return %0 : tensor<16x2xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<16x2xf32>
  %1 = sdy.all_slice [{}, {"x"}] %0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  return %1 : tensor<16x2xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion2
func.func @reduce_scatter_fusion2(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{"x"}, {"y"}] %arg0 out_sharding=<@mesh2, [{"x"}, {"y"}]> : tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"z"}, {"p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> :  tensor<16x8xf32>
  // CHECK-NEXT: return %1 : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion3
func.func @reduce_scatter_fusion3(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{"y"}, {"x"}] %arg0 out_sharding=<@mesh2, [{"y"}, {"x"}]> : tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"z"}, {"p"}] %0 out_sharding=<@mesh2, [{"y", "z"}, {"x", "p"}]> :  tensor<16x8xf32>
  // CHECK-NEXT: return %1 : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{"y", "z"}, {"x", "p"}] %0 out_sharding=<@mesh2, [{"y", "z"}, {"x", "p"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion4
func.func @reduce_scatter_fusion4(%arg0 : tensor<64x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"r"}, {}]>}) -> tensor<64x8xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{"y", "z"}, {"x"}] %arg0 out_sharding=<@mesh2, [{"r", "y", "z"}, {"x"}]> : tensor<64x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"p"}, {"q"}] %0 out_sharding=<@mesh2, [{"r", "y", "z", "p"}, {"x", "q"}]> :  tensor<64x8xf32>
  // CHECK-NEXT: return %1 : tensor<64x8xf32>
  %0 = sdy.all_reduce {"x", "y", "z"} %arg0 out_sharding=<@mesh2, [{"r"}, {}]> : tensor<64x8xf32>
  %1 = sdy.all_slice [{"y", "z", "p"}, {"x", "q"}] %0 out_sharding=<@mesh2, [{"r", "y", "z", "p"}, {"x", "q"}]> : tensor<64x8xf32>
  return %1 : tensor<64x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_missing_in_sharding
func.func @reduce_scatter_fusion_missing_in_sharding(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{"x"}, {"y"}] %arg0 out_sharding=<@mesh2, [{"x"}, {"y"}]> : tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"z"}, {"p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> :  tensor<16x8xf32>
  // CHECK-NEXT: return %1 : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_multiple_same_uses
func.func @reduce_scatter_fusion_multiple_same_uses(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{"x"}, {"y"}] %arg0 out_sharding=<@mesh2, [{"x"}, {"y"}]> : tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"z"}, {"p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> :  tensor<16x8xf32>
  // CHECK-NEXT: return %1 : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x8xf32>
  %2 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x8xf32>
  %3 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x8xf32>
  return %3 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_merge_subaxis
func.func @reduce_scatter_fusion_merge_subaxis(%arg0 : tensor<16x16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"r":(1)2}, {"q":(1)4}]>}) -> tensor<16x16xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{"r":(2)2, "y"}, {"q":(4)2}] %arg0 out_sharding=<@mesh2, [{"r":(1)4, "y"}, {"q"}]> : tensor<16x16xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"z"}, {"x"}] %0 out_sharding=<@mesh2, [{"r":(1)4, "y", "z"}, {"q", "x"}]> :  tensor<16x16xf32>
  // CHECK-NEXT: return %1 : tensor<16x16xf32>
  %0 = sdy.all_reduce {"r":(2)2, "q":(4)2, "y"} %arg0 out_sharding=<@mesh2, [{"r":(1)2}, {"q":(1)4}]> : tensor<16x16xf32>
  %1 = sdy.all_slice [{"r":(2)2, "y", "z"}, {"q":(4)2, "x"}] %0 out_sharding=<@mesh2, [{"r":(1)4, "y", "z"}, {"q", "x"}]> : tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_not_merge_subaxis
func.func @reduce_scatter_fusion_not_merge_subaxis(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"r":(2)2}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{"r":(1)2, "y"}, {}] %arg0 out_sharding=<@mesh2, [{"r":(2)2, "r":(1)2, "y"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"z"}, {"x"}] %0 out_sharding=<@mesh2, [{"r":(2)2, "r":(1)2, "y", "z"}, {"x"}]> :  tensor<16x8xf32>
  // CHECK-NEXT: return %1 : tensor<16x8xf32>
  %0 = sdy.all_reduce {"r":(1)2, "y"} %arg0 out_sharding=<@mesh2, [{"r":(2)2}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{"r":(1)2, "y", "z"}, {"x"}] %0 out_sharding=<@mesh2, [{"r":(2)2, "r":(1)2, "y", "z"}, {"x"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_with_replicated_axis
func.func @reduce_scatter_fusion_with_replicated_axis(%arg0 : tensor<16x16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"y"}, {}], replicated={"z"}>}) -> tensor<16x16xf32> {
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{}, {"x"}] %arg0 out_sharding=<@mesh2, [{"y"}, {"x"}], replicated={"z"}> : tensor<16x16xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"q"}, {"r"}] %0 out_sharding=<@mesh2, [{"y", "q"}, {"x", "r"}], replicated={"z"}> :  tensor<16x16xf32>
  // CHECK-NEXT: return %1 : tensor<16x16xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh2, [{"y"}, {}], replicated={"z"}> : tensor<16x16xf32>
  %1 = sdy.all_slice [{"q"}, {"x", "r"}] %0 out_sharding=<@mesh2, [{"y", "q"}, {"x", "r"}], replicated={"z"}> : tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_multiple_different_uses
func.func @reduce_scatter_fusion_multiple_different_uses(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x8xf32>
  // CHECK-NEXT: %2 = sdy.all_reduce {"r"} %0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %1 : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x8xf32>
  %2 = sdy.all_reduce {"r"} %0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_uses_cases_with_different_slicing_axes
func.func @reduce_scatter_fusion_uses_cases_with_different_slicing_axes(%arg0 : tensor<16x16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<16x16xf32> {
  // CHECK-NEXT:%0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x16xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x16xf32>
  // CHECK-NEXT: %2 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x16xf32>
  // CHECK-NEXT: %3 = sdy.all_slice [{"x", "r"}, {"y", "q"}] %0 out_sharding=<@mesh2, [{"x", "r"}, {"y", "q"}]> : tensor<16x16xf32>
  // CHECK-NEXT: return %3 : tensor<16x16xf32>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<16x16xf32>
  %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x16xf32>
  %2 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh2, [{"x", "z"}, {"y", "p"}]> : tensor<16x16xf32>
  %3 = sdy.all_slice [{"x", "r"}, {"y", "q"}] %0 out_sharding=<@mesh2, [{"x", "r"}, {"y", "q"}]> : tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_reduction_axes_not_matched1
func.func @reduce_scatter_fusion_reduction_axes_not_matched1(%arg0 : tensor<16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {}]>}) -> tensor<16x8xf32> {
  // CHECK-NEXT: %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{}, {"y"}] %0 out_sharding=<@mesh, [{}, {"y"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %1 : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_slice [{}, {"y"}] %0 out_sharding=<@mesh, [{}, {"y"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_reduction_axes_not_matched2
func.func @reduce_scatter_fusion_reduction_axes_not_matched2(%arg0 : tensor<64x16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{"r"}, {}]>}) -> tensor<64x16xf32> {
  // CHECK-NEXT: %0 = sdy.all_reduce {"x", "y", "z"} %arg0 out_sharding=<@mesh2, [{"r"}, {}]> : tensor<64x16xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"y", "p", "z"}, {"x", "q"}] %0 out_sharding=<@mesh2, [{"r", "y", "p", "z"}, {"x", "q"}]> : tensor<64x16xf32>
  // CHECK-NEXT: return %1 : tensor<64x16xf32>
  %0 = sdy.all_reduce {"x", "y", "z"} %arg0 out_sharding=<@mesh2, [{"r"}, {}]> : tensor<64x16xf32>
  %1 = sdy.all_slice [{"y", "p", "z"}, {"x", "q"}] %0 out_sharding=<@mesh2, [{"r", "y", "p", "z"}, {"x", "q"}]> : tensor<64x16xf32>
  return %1 : tensor<64x16xf32>
}

// CHECK-LABEL: func @reduce_scatter_fusion_no_subaxis_prefix_match
func.func @reduce_scatter_fusion_no_subaxis_prefix_match(%arg0 : tensor<64x16xf32> {sdy.sharding=#sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<64x16xf32> {
  // CHECK-NEXT: %0 = sdy.all_reduce {"r":(1)2, "x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<64x16xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{"r", "x", "z"}, {"y", "q"}] %0 out_sharding=<@mesh2, [{"r", "x", "z"}, {"y", "q"}]> : tensor<64x16xf32>
  // CHECK-NEXT: return %1 : tensor<64x16xf32>
  %0 = sdy.all_reduce {"r":(1)2, "x", "y"} %arg0 out_sharding=<@mesh2, [{}, {}]> : tensor<64x16xf32>
  %1 = sdy.all_slice [{"r", "x", "z"}, {"y", "q"}] %0 out_sharding=<@mesh2, [{"r", "x", "z"}, {"y", "q"}]> : tensor<64x16xf32>
  return %1 : tensor<64x16xf32>
}
