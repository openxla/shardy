// RUN: mpmd_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=4]>} {
  // CHECK-NEXT: mpmd.broadcast %arg0 : tensor<4x8xf32>
  %0 = mpmd.broadcast %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
