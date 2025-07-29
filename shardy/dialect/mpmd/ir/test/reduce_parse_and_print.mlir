// RUN: mpmd_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @reduce_with_none_reduction_type
func.func @reduce_with_none_reduction_type(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=4]>} {
  // CHECK-NEXT: mpmd.reduce<none> %arg0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = mpmd.reduce<none> %arg0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reduce_with_add_reduction_type
func.func @reduce_with_add_reduction_type(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=4]>} {
  // CHECK-NEXT: mpmd.reduce<add> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = mpmd.reduce<add> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
