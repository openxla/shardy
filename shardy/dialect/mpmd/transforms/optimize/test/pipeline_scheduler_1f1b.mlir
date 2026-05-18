// RUN: mpmd_opt %s -mpmd-pipeline-scheduler='must-happen-before=1F1B' | FileCheck %s

// Verifies that the 1F1B pipeline schedule reorders independent fragments on
// the same mesh so that forward fragments (no transpose count) are scheduled
// before backward fragments (transpose count > 0).

!mesh_1_tensor_2_2_f32 = !mpmd.mesh_tensor<"m1", tensor<2x2xf32>>

// CHECK-LABEL: func @schedule_1f1b_fwd_before_bwd
func.func @schedule_1f1b_fwd_before_bwd
(%arg0: !mesh_1_tensor_2_2_f32, %arg1: !mesh_1_tensor_2_2_f32)
 -> (!mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32)
 attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>> } {
  // Input order: backward (g) before forward (f).
  // Expected: 1F1B reorders forward before backward.
  // CHECK: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK: mpmd.fragment<mesh="m1", origin=["g"(1)]>
  %g = mpmd.fragment<mesh="m1", origin=["g"(1)]> (%arg1) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  %f = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) {call_counter = 0 : ui32} (%arg3: tensor<2x2xf32>) {
    %1 = stablehlo.add %arg3, %arg3 : tensor<2x2xf32>
    mpmd.return %1 : tensor<2x2xf32>
  } : (!mesh_1_tensor_2_2_f32) -> !mesh_1_tensor_2_2_f32
  return %f, %g : !mesh_1_tensor_2_2_f32, !mesh_1_tensor_2_2_f32
}
