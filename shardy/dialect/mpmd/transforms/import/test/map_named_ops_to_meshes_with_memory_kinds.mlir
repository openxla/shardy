// RUN: mpmd_opt %s -mpmd-map-named-ops-to-mpmd-ops='assignment=x@m1#pinned_host,y@m1#device,f1@m1#pinned_host' -mpmd-map-named-ops-to-mpmd-ops 2>&1 | FileCheck %s

// CHECK-LABEL: func @assign_to_pinned_host_and_device
func.func @assign_to_pinned_host_and_device(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  // CHECK-NEXT: assign {origin = "x"} {{.*}} (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">
  // CHECK-NEXT: unassign {origin = "x"} {{.*}} (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">) -> tensor<4x8xf32>
  // CHECK-NEXT: assign {origin = "y"} {{.*}} (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="device">
  // CHECK-NEXT: unassign {origin = "y"} {{.*}} (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="device">) -> tensor<4x8xf32>
  %0 = mpmd.named_tensor %arg0 name="x" : tensor<4x8xf32>
  %1 = mpmd.named_tensor %0 name="y" : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @map_nc_with_memory_kind
func.func @map_nc_with_memory_kind(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
// CHECK:      %[[ASSIGN:.*]] = mpmd.assign {origin = "f1"} %arg0
// CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%[[ASSIGN]]) (%arg1: {{.*}} {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "f1"} %[[FRAG]]
// CHECK-NEXT: return %[[UNASSIGN]]
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}
