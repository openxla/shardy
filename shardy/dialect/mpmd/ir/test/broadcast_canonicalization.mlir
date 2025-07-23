// RUN: mpmd_opt %s -canonicalize 2>&1 | FileCheck %s

// CHECK-LABEL: func @broadcast_of_broadcast
// CHECK-NEXT: %[[BROADCAST:.*]] = mpmd.broadcast %arg0 : tensor<4x8xf32>
// CHECK-NEXT: return %[[BROADCAST]] : tensor<4x8xf32>
func.func @broadcast_of_broadcast(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {

  %0 = mpmd.broadcast %arg0 : tensor<4x8xf32>
  %1 = mpmd.broadcast %0 : tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}
