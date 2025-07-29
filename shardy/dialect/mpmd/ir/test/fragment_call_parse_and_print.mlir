// RUN: mpmd_opt %s 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0 : !mesh_1_tensor) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2, "y"=4]>>,
      <"m2": <["z"=3]>>
    >} {
  // CHECK-NEXT: %[[FRAGMENT_CALL_0:.*]] = mpmd.fragment_call<mesh="m1", origin=["f"(1), "g"]> @fragment1(%arg0)
  // CHECK-NEXT: %[[FRAGMENT_CALL_1:.*]] = mpmd.fragment_call<mesh="m1", origin=[]> @fragment1(%[[FRAGMENT_CALL_0]])
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %[[FRAGMENT_CALL_1]]
  // CHECK-NEXT: %[[FRAGMENT_CALL_2:.*]] = mpmd.fragment_call<mesh="m2", origin=[]> @fragment2(%[[TRANSFER]])
  // CHECK-NEXT: return %[[FRAGMENT_CALL_2]]
  %0 = mpmd.fragment_call<mesh="m1", origin=["f"(1), "g"]> @fragment1(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.fragment_call<mesh="m1", origin=[]> @fragment1(%0) : (!mesh_1_tensor) -> !mesh_1_tensor
  %2 = mpmd.transfer %1 : (!mesh_1_tensor) -> !mesh_2_tensor
  %3 = mpmd.fragment_call<mesh="m2", origin=[]> @fragment2(%2) : (!mesh_2_tensor) -> !mesh_2_tensor
  func.return %3 : !mesh_2_tensor
}

// CHECK-LABEL: func @fragment1
func.func @fragment1(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=4]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @fragment2
func.func @fragment2(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["z"=3]>} {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
