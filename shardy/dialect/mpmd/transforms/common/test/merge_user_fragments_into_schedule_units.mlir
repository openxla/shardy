// RUN: mpmd_opt %s -mpmd-merge-user-fragments-into-scheduling-units 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @simple_merge
func.func @simple_merge(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1", "f2"]> (%arg0)
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @transpose_counts_do_not_match
func.func @transpose_counts_do_not_match(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"(1)]>
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @call_counts_do_not_match
func.func @call_counts_do_not_match(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 0 : ui32}
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[FRAGMENT]]) {call_counter = 1 : ui32}
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  %0 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 0 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @call_counts_match
func.func @call_counts_match(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 2 : ui32}
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 2 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%0) {call_counter = 2 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @producer_call_count_is_undefined
func.func @producer_call_count_is_undefined(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 2 : ui32}
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%0) {call_counter = 2 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @consumer_call_count_is_undefined
func.func @consumer_call_count_is_undefined(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 3 : ui32}
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0) {call_counter = 3 : ui32} (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%0)(%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @same_stage_can_be_merged
func.func @same_stage_can_be_merged(%arg0: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1", "f2"], stage=0> (%arg0)
  // CHECK-NOT: mpmd.fragment
  %0 = mpmd.fragment<mesh="m1", origin=["f1"], stage=0> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f2"], stage=0> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @different_stages_cannot_be_merged
func.func @different_stages_cannot_be_merged(%arg0: !mesh_1_tensor_4_8_f32)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["f1"], stage=0>
  // CHECK: mpmd.fragment<mesh="m1", origin=["f2"], stage=1>
  %0 = mpmd.fragment<mesh="m1", origin=["f1"], stage=0> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f2"], stage=1> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}
