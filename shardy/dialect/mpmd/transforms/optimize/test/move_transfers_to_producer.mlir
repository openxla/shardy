// RUN: mpmd_opt %s -mpmd-move-transfers-to-producer 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>> >

// CHECK-LABEL: func private @move_transfer_right_after_producer
func.func private @move_transfer_right_after_producer
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#topology}
{
  // CHECK-NEXT: %[[FRAGMENT1:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]>
  // CHECK-NEXT: mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %0
  // CHECK-NEXT: %[[FRAGMENT2:.*]] = mpmd.fragment<mesh="m1", origin=["another_producer"]>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["another_producer"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %transfer = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  func.return %1, %transfer : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func private @move_transfer_of_block_arg_at_beginning_of_block
func.func private @move_transfer_of_block_arg_at_beginning_of_block
  (%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#topology}
{
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0
  // CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["producer"]>
  %0 = mpmd.fragment<mesh="m1", origin=["producer"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %transfer = mpmd.transfer %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  func.return %0, %transfer : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}
