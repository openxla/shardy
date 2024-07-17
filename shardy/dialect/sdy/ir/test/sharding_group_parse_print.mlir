// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @add_to_default_group_type
func.func @add_to_default_group_type(%arg0: tensor<8xf32>) {
  // CHECK sdy.sharding_group %arg0 group_id=21 type=AS  : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>
  func.return
}
