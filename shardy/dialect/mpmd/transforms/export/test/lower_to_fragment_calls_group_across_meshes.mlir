// RUN: mpmd_opt %s -mpmd-lower-to-fragment-calls='group-across-meshes=true' -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor)
    -> (!mesh_2_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=2,"y"=2]>>
    >} {
  // Two identical fragment but with different mesh shapes.
  // CHECK-NEXT: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT0:.*]](%arg0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.sine %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor

  // CHECK-NEXT: mpmd.fragment_call<mesh="m2", origin=["f2"]> @[[FRAGMENT1:.*]](%[[TRANSFER]])
  %2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%1) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.sine %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor

  func.return %2: !mesh_2_tensor
}

// CHECK:       func @[[FRAGMENT0]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:  mesh_shape = #sdy.mesh<["x"=4]>


// CHECK:       func @[[FRAGMENT1]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-SAME:  mesh_shape = #sdy.mesh<["x"=2, "y"=2]>

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor)
    -> (!mesh_2_tensor) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["y"=4]>>
    >} {
  // Two identical fragment on different meshes with identical shapes.
  // CHECK-NEXT: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[FRAGMENT:.*]](%arg0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.sine %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor

  // CHECK-NEXT: mpmd.fragment_call<mesh="m2", origin=["f2"]> @[[FRAGMENT:.*]](%[[TRANSFER]])
  %2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%1) {xla_tpu_user_reserved_hbm_bytes = 256 : i64}
    (%arg1: tensor<4x8xf32>) {
    %3 = stablehlo.sine %arg1 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor

  func.return %2: !mesh_2_tensor
}

// CHECK:       func @[[FRAGMENT]](%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NOT:   func
