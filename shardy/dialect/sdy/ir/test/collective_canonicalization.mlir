// RUN: sdy_opt %s -canonicalize | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

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

// CHECK-LABEL: func @null_all_to_all
func.func @null_all_to_all(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.all_to_all {} 0->1 %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
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
func.func @all_slice_of_all_gather_mismatching_axes_per_dim(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK: %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x"}]> :  tensor<16x2xf32>
  // CHECK-NEXT: %1 = sdy.all_slice [{}, {"y"}] %0 out_sharding=<@mesh, [{}, {"x", "y"}]> :  tensor<16x2xf32>
  // CHECK-NEXT: return %1 : tensor<16x2xf32>
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x"}]> :  tensor<16x2xf32>
  %1 = sdy.all_slice [{}, {"y"}] %0 out_sharding=<@mesh, [{}, {"x", "y"}]> :  tensor<16x2xf32>
  return %1 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_slice_of_all_gather_many_uses
func.func @all_slice_of_all_gather_many_uses(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  // CHECK: %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  // CHECK-NEXT: return %arg0, %0 : tensor<16x2xf32>, tensor<16x2xf32>
  %0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x2xf32>
  %1 = sdy.all_slice [{}, {"x"}] %0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %arg0, %0 : tensor<16x2xf32>, tensor<16x2xf32>
}
