// RUN: mpmd_opt %s -verify-diagnostics

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=4]>} {
  %0 = mpmd.reduce<none> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32> // expected-error {{ReduceOp must have exactly one operand if the reduction type is none}}
  func.return %0 : tensor<4x8xf32>
}
