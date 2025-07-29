// RUN: mpmd_opt %s -mpmd-map-input-output-to-mesh='input-assignment=0@m#pinned_host,1@m#device output-assignment=1@m#pinned_host,2@m#device' -verify-diagnostics -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: func @map_inputs_to_mesh_with_memory_kind
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">
// CHECK-SAME: (tensor<4x8xf32>, !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">, !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">)
func.func @map_inputs_to_mesh_with_memory_kind(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2]>>>}
{
  // CHECK-DAG: mpmd.unassign {origin = "user_in"}  %arg0 : (!mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">)
  // CHECK-DAG: mpmd.unassign {origin = "user_in"}  %arg1 : (!mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">)
  // CHECK: %[[C:.*]] = stablehlo.constant
  // CHECK-DAG: mpmd.assign {origin = "user_out"}  %[[C]] {{.*}} -> !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">
  // CHECK-DAG: mpmd.assign {origin = "user_out"}  %[[C]] {{.*}} -> !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">
  %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  %1 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  func.return %0, %1, %1 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @memory_kinds_in_type_and_attr
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">
// CHECK-SAME: %arg1: !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">
// CHECK-SAME: (tensor<4x8xf32> {mhlo.memory_kind = "device"}
// CHECK-SAME: , !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">
// CHECK-SAME: , !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">)
func.func @memory_kinds_in_type_and_attr(%arg0: tensor<4x8xf32> {mhlo.memory_kind = "pinned_host"}, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32> {mhlo.memory_kind = "device"}, tensor<4x8xf32>, tensor<4x8xf32> {mhlo.memory_kind = "device"})
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2]>>>}
{
  // CHECK-DAG: mpmd.unassign {origin = "user_in"}  %arg0 : (!mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">)
  // CHECK-DAG: mpmd.unassign {origin = "user_in"}  %arg1 : (!mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">)
  // CHECK: %[[C:.*]] = stablehlo.constant
  // CHECK-DAG: mpmd.assign {origin = "user_out"}  %[[C]] {{.*}} -> !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="device">
  // CHECK-DAG: mpmd.assign {origin = "user_out"}  %[[C]] {{.*}} -> !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">
  %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  %1 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  func.return %0, %1, %1 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}
