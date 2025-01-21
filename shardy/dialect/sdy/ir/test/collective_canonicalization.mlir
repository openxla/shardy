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

// CHECK-LABEL: func @null_all_reduce
func.func @null_all_reduce(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<16x2xf32> {
  // CHECK-NEXT: return %arg0 : tensor<16x2xf32>
  %0 = sdy.all_reduce {} %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}
