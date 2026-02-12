// RUN: mpmd_opt %s -mpmd-map-named-ops-to-mpmd-ops='assignment=actual_reference@m1,n1@m1,n2@m2' -mpmd-introduce-transfers 2>&1 | FileCheck %s

sdy.mesh @m1 = <["x"=2]>
sdy.mesh @m2 = <["x"=2]>

!mesh_1_replicated = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_replicated = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @only_actual
func.func @only_actual(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "actual_reference"} %arg0
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "actual_reference"} %[[ASSIGN]]
  // CHECK-NEXT: return %[[UNASSIGN]]
  %0 = mpmd.named_tensor %arg0 name="actual_reference" : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @actual_and_unused
func.func @actual_and_unused(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "actual_reference"} %arg0
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "actual_reference"} %[[ASSIGN]]
  // CHECK-NEXT: return %[[UNASSIGN]]
  %0 = mpmd.named_tensor %arg0 name="actual_reference" : tensor<4x8xf32>
  %1 = mpmd.named_tensor %0 name="unused" : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @only_unused
func.func @only_unused(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // CHECK-NEXT: return %arg0
  %0 = mpmd.named_tensor %arg0 name="unused1" : tensor<4x8xf32>
  %1 = mpmd.named_tensor %0 name="unused2" : tensor<4x8xf32>
  %2 = mpmd.named_tensor %1 name="unused2" : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @duplicate_actual
func.func @duplicate_actual(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // CHECK:      %[[ASSIGN_1:.*]] = mpmd.assign {origin = "actual_reference"} %arg0
  // CHECK-NEXT: %[[UNASSIGN_1:.*]] = mpmd.unassign {origin = "actual_reference"} %[[ASSIGN_1]]
  // CHECK-NEXT: return %[[UNASSIGN_1]]
  %0 = mpmd.named_tensor %arg0 name="actual_reference" : tensor<4x8xf32>
  %1 = mpmd.named_tensor %0 name="actual_reference" : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @named_tensor_transfer
func.func @named_tensor_transfer(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  // CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "n1"} %arg0
  // CHECK-NEXT: %[[TRANSFER_0:.*]] = mpmd.transfer %[[ASSIGN]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: %[[TRANSFER_1:.*]] = mpmd.transfer %[[TRANSFER_0]] : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "n1"} %[[TRANSFER_1]]
  // CHECK-NEXT: return %[[UNASSIGN]]
  %0 = mpmd.named_tensor %arg0 name="n1" : tensor<4x8xf32>
  %1 = mpmd.named_tensor %0 name="n2" : tensor<4x8xf32>
  %2 = mpmd.named_tensor %1 name="n1" : tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}
