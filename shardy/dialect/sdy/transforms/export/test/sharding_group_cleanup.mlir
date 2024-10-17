// RUN: sdy_opt %s -sdy-sharding-group-cleanup | FileCheck %s

// CHECK-LABEL: func @func_with_sharding_group_ops_has_them_removed
func.func @func_with_sharding_group_ops_has_them_removed(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<32x96xf32>
  // CHECK:   return %1 : tensor<32x96xf32>
  sdy.sharding_group %arg0 group_id = 747 : tensor<32x96xf32>
  sdy.sharding_group %0 group_id = 747 : tensor<32x96xf32>
  sdy.sharding_group %1 group_id = 747 : tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}
