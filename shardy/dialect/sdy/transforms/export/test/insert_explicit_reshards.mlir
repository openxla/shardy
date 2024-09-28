// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @dot_matrix_matrix_compatible
func.func @dot_matrix_matrix_compatible(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding_per_value = #sdy.sharding<@mesh, [{"x"}, {}]>}
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding_per_value = #sdy.sharding<@mesh, [{"x"}, {}]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}


// CHECK-LABEL: func @dot_matrix_matrix_incompatible_same_non_contracting_dims
func.func @dot_matrix_matrix_incompatible_same_non_contracting_dims(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<8x16xf32> {
  // CHECK: stablehlo.dot %arg0, %arg1 {sdy.sharding_per_value = #sdy.sharding<@mesh, [{"x"}, {}]>}
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding_per_value = #sdy.sharding<@mesh, [{"x"}, {}]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
