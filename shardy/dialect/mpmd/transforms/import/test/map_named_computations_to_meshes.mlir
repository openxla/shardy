// RUN: mpmd_opt %s -mpmd-map-named-ops-to-mpmd-ops='assignment=f1@m1,f2@m1' 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @simple_assignment
func.func @simple_assignment(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
// CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "f1"}  %arg0
// CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%[[ASSIGN]]) (%arg1: {{.*}} {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "f1"}  %[[FRAG]]
// CHECK-NEXT: return %[[UNASSIGN]]
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @simple_assignment_transpose
func.func @simple_assignment_transpose(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
// CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "f1"}  %arg0
// CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%[[ASSIGN]]) (%arg1: {{.*}} {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "f1"}  %[[FRAG]]
// CHECK-NEXT: return %[[UNASSIGN]]
  %1 = mpmd.named_computation<"f1"(1)> (%arg0) (%arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}
