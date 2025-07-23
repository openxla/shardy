// RUN: mpmd_opt %s -mpmd-merge-forward-with-backward 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

// CHECK-LABEL: func @merge_forward_producer_with_backward_consumer
func.func @merge_forward_producer_with_backward_consumer(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) attributes {
    "topology"=#topology} {
  // CHECK: mpmd.fragment<mesh="m2", origin=["f2"]> (%arg1)
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1", "f1"(1)]> (%arg0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // This fragment is on a different mesh and should not block merging.
  %2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%arg1)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1, %2 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @do_not_merge_if_consumer_fragment_not_backward
func.func @do_not_merge_if_consumer_fragment_not_backward(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#topology} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  // CHECK: mpmd.fragment<mesh="m1", origin=["f2"]> (%0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @do_not_merge_if_producer_fragment_not_forward
func.func @do_not_merge_if_producer_fragment_not_forward(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#topology} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0)
  // CHECK: mpmd.fragment<mesh="m1", origin=["f2"]> (%0)
  %0 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @do_not_merge_if_consumer_not_immediately_after_producer
func.func @do_not_merge_if_consumer_not_immediately_after_producer(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
   "topology"=#topology} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK: mpmd.fragment<mesh="m1", origin=["f2"]>
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"(1)]>
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // This fragment is on the same mesh which makes consumer not immediately after producer.
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%0)
    (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1, %2 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
