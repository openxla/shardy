// RUN: mpmd_opt %s -mpmd-lower-to-fragment-calls -split-input-file 2>&1 | FileCheck %s

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4x8xf32>>
// CHECK-LABEL: func @guid_printing
func.func @guid_printing(%arg0: !mesh_tensor) -> !mesh_tensor
  attributes {"topology"=#mpmd.topology<<"m": <["x"=4]>>>} {
  // CHECK: mpmd.fragment_call<mesh="m", origin=["block"]> @p0_block_fwd.main
  %0 = mpmd.fragment<mesh="m", origin=["block"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  return %0 : !mesh_tensor
}

// -----
!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4x8xf32>>
// CHECK-LABEL: func @one_guid_multiple_call_sites
func.func @one_guid_multiple_call_sites(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
  attributes {"topology"=#mpmd.topology<<"m": <["x"=4]>>>} {
  // CHECK: mpmd.fragment_call<mesh="m", origin=["foo"]> @p0_foo_fwd.main
  %0 = mpmd.fragment<mesh="m", origin=["foo"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  // This is reusing the same fragment call as above.
  // CHECK: mpmd.fragment_call<mesh="m", origin=["bar"]> @p0_foo_fwd.main
  %1 = mpmd.fragment<mesh="m", origin=["bar"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  return %0, %1 : !mesh_tensor, !mesh_tensor
}

// -----
!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4x8xf32>>
// CHECK-LABEL: func @multiple_guids
func.func @multiple_guids(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
  attributes {"topology"=#mpmd.topology<<"m": <["x"=4]>>>} {
  // CHECK: mpmd.fragment_call<mesh="m", origin=["foo"]> @p0_foo_fwd.main
  %0 = mpmd.fragment<mesh="m", origin=["foo"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64} (%arg1: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  // CHECK: mpmd.fragment_call<mesh="m", origin=["bar"(1)]> @p1_bar_bwd.main
  %1 = mpmd.fragment<mesh="m", origin=["bar"(1)]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  return %0, %1 : !mesh_tensor, !mesh_tensor
}

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4x8xf32>>

// CHECK-LABEL: module @jit_test_module
module @jit_test_module {
// CHECK-LABEL: func @custom_module_name
func.func @custom_module_name(%arg0: !mesh_tensor) -> !mesh_tensor
  attributes {"topology"=#mpmd.topology<<"m": <["x"=4]>>>} {
  // CHECK: mpmd.fragment_call<mesh="m", origin=["block"]> @p0_block_fwd.test_module
  %0 = mpmd.fragment<mesh="m", origin=["block"]> (%arg0) {xla_tpu_user_reserved_hbm_bytes = 256 : i64} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  return %0 : !mesh_tensor
}
}
