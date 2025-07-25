// RUN: mpmd_opt %s -mpmd-remat-fragment='merge-remat-fragments=true'  2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @merge_remat_into_bwd_consumers
func.func @merge_remat_into_bwd_consumers(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {

  // CHECK-NEXT: %[[FORWARD:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  // CHECK-NEXT:   stablehlo.add
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %forward = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {call_counter = 1 : ui32} (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // This fragment is unused, but necessary to guarantee that (%forward,
  // %backward) form a candidate pair for remat, i.e., they cannot be adjacent
  // in the program.
  // CHECK-NEXT: %[[_:.*]] = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0)
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %in_between = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 1 : ui32} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // The forward fragment is rematted and merged into its first backward user.
  // CHECK-NEXT: %[[BACKWARD:.*]] = mpmd.fragment<mesh="m1", origin=["f1", "f1"(1)]> (%arg0) {remat} (%arg1: tensor<4x8xf32>)
  // CHECK-NEXT:   add
  // CHECK-NEXT:   subtract
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %backward = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward) {call_counter = 1 : ui32} (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.subtract %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // The forward fragment is rematted and merged into its second backward user.
  // CHECK-NEXT: %[[ANOTHER_BACKWARD:.*]] = mpmd.fragment<mesh="m1", origin=["f1", "f1"(1)]> (%arg0) {remat} (%arg1: tensor<4x8xf32>)
  // CHECK-NEXT:   add
  // CHECK-NEXT:   multiply
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %another_backward = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward) {call_counter = 1 : ui32} (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.multiply %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // CHECK-NEXT: %[[FORWARD]], %[[BACKWARD]], %[[ANOTHER_BACKWARD]]
  return %forward, %backward, %another_backward : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
