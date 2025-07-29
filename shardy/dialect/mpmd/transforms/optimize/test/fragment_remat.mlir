// RUN: mpmd_opt %s -mpmd-remat-fragment 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @should_remat_if_all_conditions_met
func.func @should_remat_if_all_conditions_met(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // Forward fragment's result is used by the backward fragment below and
  // they have matching call_counters. So we should insert a remat before the backward fragment.
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  // CHECK-NEXT: mpmd.return
  // CHECK-NEXT: }
  %forward_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // Another fragment is between the forward fragment above and the backward fragment below. The remat fragment should be inserted right before the backward fragment.
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0)
  // CHECK-NEXT: mpmd.return
  // CHECK-NEXT: }
  %another_forward_result = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 2 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // Backward fragment.
  // CHECK-NEXT: %[[REMAT_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {call_counter = 1 : ui32, remat}
  // CHECK-NEXT: mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[REMAT_RESULT]])
  %backward_result = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward_result) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  return %backward_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_not_remat_if_result_of_forward_not_used_by_backward_fragment
func.func @should_not_remat_if_result_of_forward_not_used_by_backward_fragment(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // Forward fragment's result is not used by the backward fragment below.
  // So no remat should happen.
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  %forward_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // A fragment that goes between the forward and backward fragments.
  %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // Backward fragment.
  // CHECK: mpmd.fragment<mesh="m1", origin=["f2"(1)]> (%arg0)
  // CHECK-NOT: {remat}
  %backward_result = mpmd.fragment<mesh="m1", origin=["f2"(1)]> (%arg0) {call_counter = 1 : ui32}  (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  return %backward_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_not_remat_if_call_counters_mismatch
func.func @should_not_remat_if_call_counters_mismatch(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // The call_counter of the forward and backward operations are different.
  // So no remat should happen.
  // CHECK: %[[FORWARD_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  %forward_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32}  (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // A fragment that goes between the forward and backward fragments.
  %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // Backward fragment.
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[FORWARD_RESULT]])
  // CHECK-NOT: {remat}
  %backward_result = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward_result) {call_counter = 100 : ui32}  (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  return %backward_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_not_remat_if_not_backward_fragment
func.func @should_not_remat_if_not_backward_fragment(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // The second fragment is not a backward fragment (transpose count is not 1).
  // So no remat should happen.
  // CHECK: %[[FORWARD_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  %forward_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32}  (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"(100)]> (%[[FORWARD_RESULT]])
  // CHECK-NOT: {remat}
  %non_backward_result = mpmd.fragment<mesh="m1", origin=["f1"(100)]> (%forward_result) {call_counter = 100 : ui32}  (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  return %non_backward_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_not_remat_if_backward_immediately_after_forward
func.func @should_not_remat_if_backward_immediately_after_forward(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // Forward fragment.
  // CHECK: %[[FORWARD_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  %forward_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // Backward fragment.
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[FORWARD_RESULT]])
  // CHECK-NOT: {remat}
  %backward_result = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward_result) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  return %backward_result : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @should_remat_both_if_two_matching_backward_fragments
func.func @should_remat_both_if_two_matching_backward_fragments(%arg0: !mesh_1_tensor_4_8_f32)
-> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // Forward fragment.
  // CHECK: %[[FORWARD_RESULT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  %forward_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
     mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // A fragment that goes between the forward and backward fragments.
  %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // There are two backward fragments matching the forward fragment.
  // We should do remat for both.
  // CHECK: %[[REMAT_RESULT1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {call_counter = 1 : ui32, remat}
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[REMAT_RESULT1]])
  %backward_result_0 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward_result) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // CHECK: %[[REMAT_RESULT2:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {call_counter = 1 : ui32, remat}
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[REMAT_RESULT2]])
  %backward_result_1 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward_result) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.abs %arg2: tensor<4x8xf32>
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
 return %backward_result_0, %backward_result_1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32}

// CHECK-LABEL: func @should_only_remat_once_if_multiple_results_of_forward_used_by_backward
func.func @should_only_remat_once_if_multiple_results_of_forward_used_by_backward(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // Both of forward fragment's results is used by the backward fragment below.
  // We should only insert one remat fragment before the backward fragment.
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
  // CHECK-NEXT: stablehlo.add
  // CHECK-NEXT: mpmd.return
  // CHECK-NEXT: }
  %forward_result_0, %forward_result_1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %arg2, %2 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  // A fragment that goes between the forward and backward fragments. It should be kept as it is.
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0) {call_counter = 1 : ui32}
  // CHECK-NEXT: mpmd.return
  // CHECK-NEXT: }
  %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  // Backward fragment. Remat should be inserted before this.
  // CHECK-NEXT: %[[REMAT_RESULT:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {call_counter = 1 : ui32, remat}
  // CHECK-NEXT: stablehlo.add
  // CHECK-NEXT: mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[REMAT_RESULT]]#0, %[[REMAT_RESULT]]#1) {call_counter = 1 : ui32}
  %backward_result = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%forward_result_0, %forward_result_1) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  return %forward_result_1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @different_stages_mean_no_remat
func.func @different_stages_mean_no_remat(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0) {call_counter = 1 : ui32}
  // CHECK: mpmd.fragment<mesh="m1", origin=["f"(1)], stage=1> (%0) {call_counter = 1 : ui32}
  %0 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0) {call_counter = 1 : ui32} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"(1)], stage=1> (%0) {call_counter = 1 : ui32} (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}
